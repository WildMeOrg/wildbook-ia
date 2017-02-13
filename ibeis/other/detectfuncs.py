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
    label = label.replace('giraffe_reticulated', 'GIR')
    label = label.replace('zebra_grevys',        'GZ')
    label = label.replace('giraffe_masai',       'GIRM')
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
    return label


@register_ibs_method
def export_to_xml(ibs, offset='auto', enforce_yaw=False, target_size=900, purge=False,
                  use_maximum_linear_dimension=True, use_existing_train_test=True, **kwargs):
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

    train_gid_set = set(general_get_imageset_gids(ibs, 'TRAIN_SET'), **kwargs)
    test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET'), **kwargs)

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
                species_name = ibs.get_annot_species_texts(aid)
                yaw = ibs.get_annot_yaws(aid)
                if yaw != -1 and yaw is not None:
                    info['pose'] = '%0.06f' % (yaw, )
                else:
                    yawed = False
                    print("UNVIEWPOINTED: %d " % gid)
                annotation.add_object(
                    species_name, (xmax, xmin, ymax, ymin), **info)
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
            viewpoint = ibs.get_annot_yaw_texts(aid)
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
        for viewpoint in const.VIEWTEXT_TO_YAW_RADIANS:
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


def general_precision_recall_algo(ibs, label_list, confidence_list, category='positive', samples=10000, **kwargs):
    def errors(zipped, conf):
        error_list = [0, 0, 0, 0]
        for index, (label, confidence) in enumerate(zipped):
            if label == category and conf <= confidence:
                error_list[0] += 1
            elif label != category and conf <= confidence:
                error_list[2] += 1
            elif label == category:
                error_list[3] += 1
            elif label != category:
                error_list[1] += 1
        return error_list

    zipped = list(zip(label_list, confidence_list))
    conf_list = [ _ / float(samples) for _ in range(0, int(samples) + 1) ]
    conf_dict = {}
    for conf in conf_list:
        conf_dict[conf] = errors(zipped, conf)

    conf_list_ = [-1.0]
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
        conf_list_.append(conf)
        pr_list.append(pr)
        re_list.append(re)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return conf_list_, pr_list, re_list, tpr_list, fpr_list


def general_identify_operating_point(conf_list, x_list, y_list, invert=False, x_limit=0.9):
    best_length = np.inf
    best_conf_list = []
    best_x_list = []
    best_y_list = []
    for conf, x, y in zip(conf_list, x_list, y_list):
        x_ = 1.0 - x if not invert else x
        if x_ > 1.0 - x_limit:
            continue
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


def general_area_best_conf(conf_list, x_list, y_list, label='Unknown', color='b', invert=False, x_limit=0.9, **kwargs):
    import matplotlib.pyplot as plt
    best_conf_list, best_x_list, best_y_list = general_identify_operating_point(conf_list, x_list, y_list, invert=invert, x_limit=0.0)
    best_conf = best_conf_list[0]
    # best_conf_list_ = ','.join([ '%0.02f' % (conf, ) for conf in best_conf_list ])
    # label = '%s [OP = %s]' % (label, best_conf_list_, )
    label = '%s [OP = %0.02f]' % (label, best_conf, )
    linestyle = '--' if kwargs.get('classify', False) else '-'
    plt.plot(x_list, y_list, color=color, linestyle=linestyle, label=label)
    plt.plot(best_x_list, best_y_list, color=color, marker='o')
    area = np.trapz(y_list, x=x_list)
    if len(best_conf_list) > 1:
        print('WARNING: %r' % (best_conf_list, ))
    tup = general_identify_operating_point(conf_list, x_list, y_list, x_limit=x_limit)
    return area, best_conf, tup


def general_confusion_matrix_algo(label_correct_list, label_predict_list,
                                  category_list, category_mapping,
                                  fig_, axes_, fuzzy_dict=None, conf=None,
                                  conf_list=None, **kwargs):
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
                verticalalignment='center'
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


def general_tp_fp_fn(gt_list, pred_list, min_overlap, duplicate_assign=True,
                     check_species=False, check_viewpoint=False, **kwargs):
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
                # print(gt_list[gt]['species'], pred_list[pred]['species'])
                # print(gt_list[gt]['viewpoint'], pred_list[pred]['viewpoint'])
                if gt_list[gt]['species'] != pred_list[pred]['species']:
                    tp -= 1
                elif check_viewpoint and gt_list[gt]['viewpoint'] != pred_list[pred]['viewpoint']:
                    tp -= 1
        fp = num_pred - tp
        fn = num_gt - tp
    return tp, fp, fn


def general_get_imageset_gids(ibs, imageset_text, species_set=None, unique=False,
                              **kwargs):
    imageset_id = ibs.get_imageset_imgsetids_from_text(imageset_text)
    test_gid_list = ibs.get_imageset_gids(imageset_id)
    if species_set is not None:
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
                'xtl'        : bbox[0] / width,
                'ytl'        : bbox[1] / height,
                'xbr'        : (bbox[0] + bbox[2]) / width,
                'ybr'        : (bbox[1] + bbox[3]) / height,
                'width'      : bbox[2] / width,
                'height'     : bbox[3] / height,
                'species'    : ibs.get_annot_species_texts(aid),
                'viewpoint'  : ibs.get_annot_yaw_texts(aid),
                'confidence' : 1.0,
            }
            gt_list.append(temp)
        gt_dict[uuid] = gt_list
    return gt_dict


def _get_localizations(depc, gid_list, algo, config_filepath=None, classifier_masking=False, **kwargs):
    config1 = {'algo': algo, 'config_filepath': config_filepath}
    config2 = {'algo': algo, 'config_filepath': config_filepath, 'classifier_masking': classifier_masking}
    # depc.delete_property('localizations_classifier', gid_list, config=config)
    return [
        depc.get_property('localizations', gid_list, 'score',   config=config1),
        depc.get_property('localizations', gid_list, 'bboxes',  config=config1),
        depc.get_property('localizations', gid_list, 'thetas',  config=config1),
        depc.get_property('localizations', gid_list, 'confs',   config=config1),
        depc.get_property('localizations', gid_list, 'classes', config=config1),
        depc.get_property('localizations_classifier', gid_list, 'class', config=config2),
        depc.get_property('localizations_classifier', gid_list, 'score', config=config2),
    ]


def _get_all_localizations(depc, gid_list, **kwargs):

    metadata = {}

    limited = kwargs.get('limited', False)

    # Get Localizations
    if limited:
        metadata['YOLO2']  = _get_localizations(depc, gid_list, 'darknet', 'pretrained-v2-large-pascal', **kwargs)
    else:
        metadata['YOLO1']  = _get_localizations(depc, gid_list, 'darknet', 'pretrained-v2-pascal', **kwargs)
        metadata['YOLO2']  = _get_localizations(depc, gid_list, 'darknet', 'pretrained-v2-large-pascal', **kwargs)
        metadata['YOLO3']  = _get_localizations(depc, gid_list, 'darknet', 'pretrained-tiny-pascal', **kwargs)

    # metadata['SS1']    = _get_localizations(depc, gid_list, 'selective-search', **kwargs)
    # metadata['SS2']    = _get_localizations(depc, gid_list, 'selective-search-rcnn', **kwargs)

    if limited:
        metadata['FRCNN1'] = _get_localizations(depc, gid_list, 'faster-rcnn', 'pretrained-vgg-pascal', **kwargs)
    else:
        metadata['FRCNN1'] = _get_localizations(depc, gid_list, 'faster-rcnn', 'pretrained-vgg-pascal', **kwargs)
        metadata['FRCNN2'] = _get_localizations(depc, gid_list, 'faster-rcnn', 'pretrained-zf-pascal', **kwargs)

    if limited:
        metadata['SSD4']   = _get_localizations(depc, gid_list, 'ssd', 'pretrained-512-pascal-plus', **kwargs)
    else:
        metadata['SSD1']   = _get_localizations(depc, gid_list, 'ssd', 'pretrained-300-pascal', **kwargs)
        metadata['SSD2']   = _get_localizations(depc, gid_list, 'ssd', 'pretrained-512-pascal', **kwargs)
        metadata['SSD3']   = _get_localizations(depc, gid_list, 'ssd', 'pretrained-300-pascal-plus', **kwargs)
        metadata['SSD4']   = _get_localizations(depc, gid_list, 'ssd', 'pretrained-512-pascal-plus', **kwargs)

    # Get Combined
    metadata['COMBINED'] = []
    for key in metadata:
        if len(metadata['COMBINED']) == 0:
            # Initializing combined list, simply append
            metadata['COMBINED'] = list(metadata[key])
        else:
            # Combined already initialized, hstack new metadata
            current = metadata['COMBINED']
            detect = metadata[key]
            for index in range(len(current)):
                # print(index, current[index].shape, detect[index].shape)
                new = []
                for image in range(len(detect[index])):
                    # print(current[index][image].shape, detect[index][image].shape)
                    if index == 0:
                        temp = 0.0
                    elif len(current[index][image].shape) == 1:
                        temp = np.hstack((current[index][image], detect[index][image]))
                    else:
                        temp = np.vstack((current[index][image], detect[index][image]))
                    new.append(temp)
                metadata['COMBINED'][index] = np.array(new)

    metadata['COMBINED'] = [
        list(zip(*metadata['COMBINED'][:5])),
        metadata['COMBINED'][5],
        metadata['COMBINED'][6],
    ]

    return metadata


def localizer_parse_pred(ibs, test_gid_list=None, **kwargs):
    depc = ibs.depc_image

    if test_gid_list is None:
        test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
    uuid_list = ibs.get_image_uuids(test_gid_list)

    # Get bounding boxes
    if kwargs.get('algo', None) == '_COMBINED':
        metadata = _get_all_localizations(depc, test_gid_list, **kwargs)
        results_list = metadata['COMBINED'][0]
    else:
        results_list = depc.get_property('localizations', test_gid_list, None, config=kwargs)

    def _compute_conf(conf, pred_, conf_):
        conf_ = conf_ if pred_ == 'positive' else 1.0 - conf_
        p = kwargs.get('p', None)
        if p is None:
            conf_ = max(0.0, conf_)
            val = conf_ * conf
        else:
            val = p * conf_ + (1.0 - p) * conf
        val = min(1.0, max(0.0, val))
        return val

    # Get confidences for boxes
    confidences_list = [
        result_list[3]
        for result_list in results_list
    ]
    if kwargs.get('classify', False):
        # Get the new confidences
        if kwargs.get('algo', None) == '_COMBINED':
            # metadata = _get_all_localizations(depc, test_gid_list)  # ALREADY HAVE METADATA
            predictions_list_ = metadata['COMBINED'][1]
            confidences_list_ = metadata['COMBINED'][2]
        else:
            predictions_list_ = depc.get_property('localizations_classifier', test_gid_list, 'class', config=kwargs)
            confidences_list_ = depc.get_property('localizations_classifier', test_gid_list, 'score', config=kwargs)
        # Compute new confidences
        zipped = zip(confidences_list, predictions_list_, confidences_list_)
        confidences_list = [
            np.array([
                _compute_conf(confidence, prediction_, confidence_ )
                for confidence, prediction_, confidence_ in zip(confidence_list, prediction_list_, confidence_list_)
            ]) for confidence_list, prediction_list_, confidence_list_ in zipped
        ]

    # Apply NMS
    if kwargs.get('nms', False):
        nms_thresh = kwargs.get('nms_thresh', 0.2)
        print('Filtering with nms_thresh = %0.02f' % (nms_thresh, ))
        count_old = 0
        count_new = 0
        keeps_list = []
        for result_list, confidence_list in zip(results_list, confidences_list):
            bbox_list = result_list[1]
            score_list = confidence_list.reshape((-1, 1))
            dets_list = np.hstack((bbox_list, score_list))
            keep_indices_list = nms(dets_list, nms_thresh)
            count_old += len(dets_list)
            count_new += len(keep_indices_list)
            keep_indices_set = set(keep_indices_list)
            keep_list = [ index in keep_indices_set for index in range(len(dets_list)) ]
            keeps_list.append(keep_list)
        count_diff = count_old - count_new
        args = (count_old, count_new, count_diff, 100.0 * count_diff / count_old, )
        print('[nms] %d old -> %d new (%d, %0.02f%% suppressed)' % args)
    else:
        keeps_list = [
            [True] * len(confidence_list)
            for confidence_list in confidences_list
        ]

    conf_thresh = kwargs.get('conf_thresh', 0.0)
    if conf_thresh > 0.0:
        print('Filtering with conf_thresh = %0.02f' % (conf_thresh, ))
    # species_set = kwargs.get('species_set', None)
    size_list = ibs.get_image_sizes(test_gid_list)
    zipped_list = zip(results_list)
    # Reformat results for json
    zipped = zip(keeps_list, confidences_list, size_list, zipped_list)
    results_list = [
        [
            {
                'xtl'        : bbox[0] / width,
                'ytl'        : bbox[1] / height,
                'xbr'        : (bbox[0] + bbox[2]) / width,
                'ybr'        : (bbox[1] + bbox[3]) / height,
                'width'      : bbox[2] / width,
                'height'     : bbox[3] / height,
                'theta'      : theta,  # round(theta, 4),
                'confidence' : conf_,   # round(conf, 4),
                # 'class'      : class_,
                'species'    : class_,
            }
            for keep_, conf_, bbox, theta, conf, class_ in zip(keep_list_, confidence_list_, *zipped_[0][1:])
            if keep_ and conf_ >= conf_thresh
            # if species_set is None or class_ in species_set
        ]
        for keep_list_, confidence_list_, (width, height), zipped_ in zipped
    ]

    pred_dict = {
        uuid_ : result_list
        for uuid_, result_list in zip(uuid_list, results_list)
    }
    return pred_dict


def localizer_precision_recall_algo(ibs, samples=500, force_serial=True, **kwargs):
    test_gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **kwargs)
    uuid_list = ibs.get_image_uuids(test_gid_list)

    print('\tGather Ground-Truth')
    gt_dict = general_parse_gt(ibs, test_gid_list=test_gid_list, **kwargs)

    print('\tGather Predictions')
    pred_dict = localizer_parse_pred(ibs, test_gid_list=test_gid_list, **kwargs)

    print('\tGenerate Curves...')
    conf_list = [ _ / float(samples) for _ in range(0, int(samples) + 1) ]
    conf_list = sorted(conf_list, reverse=True)

    uuid_list_list = [ uuid_list for _ in conf_list ]
    gt_dict_list   = [ gt_dict   for _ in conf_list ]
    pred_dict_list = [ pred_dict for _ in conf_list ]
    kwargs_list    = [ kwargs    for _ in conf_list ]
    arg_iter = zip(conf_list, uuid_list_list, gt_dict_list, pred_dict_list, kwargs_list)
    pr_re_gen = ut.generate(localizer_precision_recall_algo_worker, arg_iter,
                            nTasks=len(conf_list), ordered=True,
                            chunksize=50, force_serial=force_serial)

    conf_list_ = [-1.0]
    pr_list = [1.0]
    re_list = [0.0]
    for conf, pr, re in pr_re_gen:
        conf_list_.append(conf)
        pr_list.append(pr)
        re_list.append(re)

    print('...complete')
    return conf_list_, pr_list, re_list


def localizer_precision_recall_algo_worker(tup):
    conf, uuid_list, gt_dict, pred_dict, kwargs = tup
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
        pr = 1.0
        re = 0.0
    return (conf, pr, re)


def nms(dets, thresh, use_cpu=True):
    # Interface into Faster R-CNN's Python native NMS algorithm by Girshick et al.
    from ibeis.algo.detect.nms.py_cpu_nms import py_cpu_nms
    return py_cpu_nms(dets, thresh)


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
                # print('Processing gid %r for localizer confusion matrix' % (test_gid, ))
                # ut.embed()
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
                                         category_mapping, **kwargs)


@register_ibs_method
def localizer_precision_recall_algo_display(ibs, min_overlap=0.5, figsize=(24, 7),
                                            write_images=False, min_recall=0.9, **kwargs):
    import matplotlib.pyplot as plt
    import plottool as pt

    fig_ = plt.figure(figsize=figsize)

    axes_ = plt.subplot(131)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall (Ground-Truth IOU >= %0.02f)' % (min_overlap, ))
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])

    config_list = [
        # {'label': 'V1',             'grid' : False, 'config_filepath' : 'v1', 'weight_filepath' : 'v1'},
        # {'label': 'V1 (GRID)',      'grid' : True,  'config_filepath' : 'v1', 'weight_filepath' : 'v1'},
        # {'label': 'V2',             'grid' : False, 'config_filepath' : 'v2', 'weight_filepath' : 'v2'},
        # {'label': 'V2 (GRID)',      'grid' : True,  'config_filepath' : 'v2', 'weight_filepath' : 'v2'},
        # {'label': 'V3',             'grid' : False, 'config_filepath' : 'v3', 'weight_filepath' : 'v3'},
        # {'label': 'V3 (GRID)',      'grid' : True,  'config_filepath' : 'v3', 'weight_filepath' : 'v3'},
        # {'label': 'V3 Whale Shark', 'grid' : False, 'config_filepath' : 'v3', 'weight_filepath' : 'v3', 'species_set' : set(['whale_shark'])},
        # {'label': 'V3 Whale Fluke', 'grid' : True,  'config_filepath' : 'v3', 'weight_filepath' : 'v3', 'species_set' : set(['whale_fluke'])},
        # {'label': 'LYNX',           'grid' : False, 'config_filepath' : 'lynx', 'weight_filepath' : 'lynx'},
        # {'label': 'LYNX (GRID)',    'grid' : True,  'config_filepath' : 'lynx', 'weight_filepath' : 'lynx'},

        # {'label': 'SS1', 'algo': 'selective-search', 'grid': False, 'species_set' : set(['zebra'])},
        # {'label': 'SS2', 'algo': 'selective-search-rcnn', 'grid': False, 'species_set' : set(['zebra'])},

        {'label': 'YOLO1', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : set(['zebra'])},
        {'label': 'YOLO1*', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : set(['zebra']), 'classify': True},
        {'label': 'YOLO1^', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : set(['zebra']), 'classify': True, 'classifier_masking': True},
        {'label': 'YOLO1^ 0.0', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.0, 'classifier_masking': True},
        {'label': 'YOLO1^ 0.1', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.1, 'classifier_masking': True},
        {'label': 'YOLO1^ 0.3', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.3, 'classifier_masking': True},
        {'label': 'YOLO1^ 0.5', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.5, 'classifier_masking': True},
        {'label': 'YOLO1^ 0.7', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.7, 'classifier_masking': True},
        {'label': 'YOLO1^ 0.9', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.9, 'classifier_masking': True},
        {'label': 'YOLO1^ 1.0', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : set(['zebra']), 'classify': True, 'p': 1.0, 'classifier_masking': True},

        # # {'label': 'YOLO1', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : set(['zebra'])},
        # {'label': 'YOLO2', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-large-pascal', 'species_set' : set(['zebra'])},
        # # {'label': 'YOLO3', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-tiny-pascal', 'species_set' : set(['zebra'])},
        # {'label': 'FRCNN1', 'algo': 'faster-rcnn', 'grid': False, 'config_filepath': 'pretrained-vgg-pascal', 'species_set' : set(['zebra'])},
        # # {'label': 'FRCNN2', 'algo': 'faster-rcnn', 'grid': False, 'config_filepath': 'pretrained-zf-pascal', 'species_set' : set(['zebra'])},
        # # {'label': 'SSD1', 'algo': 'ssd', 'grid': False, 'config_filepath': 'pretrained-300-pascal', 'species_set' : set(['zebra'])},
        # # {'label': 'SSD2', 'algo': 'ssd', 'grid': False, 'config_filepath': 'pretrained-512-pascal', 'species_set' : set(['zebra'])},
        # # {'label': 'SSD3', 'algo': 'ssd', 'grid': False, 'config_filepath': 'pretrained-300-pascal-plus', 'species_set' : set(['zebra'])},
        # {'label': 'SSD4', 'algo': 'ssd', 'grid': False, 'config_filepath': 'pretrained-512-pascal-plus', 'species_set' : set(['zebra'])},
        # {'label': 'COMBINED', 'algo': '_COMBINED', 'species_set' : set(['zebra'])},
        # {'label': 'COMBINED* ~0.1', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'nms': True, 'nms_thresh': 0.1},

        # {'label': 'COMBINED`', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'limited': True},
        # {'label': 'COMBINED`* ~0.1', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'nms': True, 'nms_thresh': 0.1, 'limited': True},

        # {'label': 'COMBINED !0.1', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'conf_thresh': 0.1},
        # {'label': 'COMBINED !0.5', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'conf_thresh': 0.5},
        # {'label': 'COMBINED !0.9', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'conf_thresh': 0.9},
        # {'label': 'COMBINED ~0.1', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'nms': True, 'nms_thresh': 0.1},
        # {'label': 'COMBINED ~0.5', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'nms': True, 'nms_thresh': 0.5},
        # {'label': 'COMBINED ~0.9', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'nms': True, 'nms_thresh': 0.9},

        # # {'label': 'YOLO1*', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-pascal', 'species_set' : set(['zebra']), 'classify': True},
        # {'label': 'YOLO2*', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-v2-large-pascal', 'species_set' : set(['zebra']), 'classify': True},
        # # {'label': 'YOLO3*', 'algo': 'darknet', 'grid': False, 'config_filepath': 'pretrained-tiny-pascal', 'species_set' : set(['zebra']), 'classify': True},
        # {'label': 'FRCNN1*', 'algo': 'faster-rcnn', 'grid': False, 'config_filepath': 'pretrained-vgg-pascal', 'species_set' : set(['zebra']), 'classify': True},
        # # {'label': 'FRCNN2*', 'algo': 'faster-rcnn', 'grid': False, 'config_filepath': 'pretrained-zf-pascal', 'species_set' : set(['zebra']), 'classify': True},
        # # {'label': 'SSD1*', 'algo': 'ssd', 'grid': False, 'config_filepath': 'pretrained-300-pascal', 'species_set' : set(['zebra']), 'classify': True},
        # # {'label': 'SSD2*', 'algo': 'ssd', 'grid': False, 'config_filepath': 'pretrained-512-pascal', 'species_set' : set(['zebra']), 'classify': True},
        # # {'label': 'SSD3*', 'algo': 'ssd', 'grid': False, 'config_filepath': 'pretrained-300-pascal-plus', 'species_set' : set(['zebra']), 'classify': True},
        # {'label': 'SSD4*', 'algo': 'ssd', 'grid': False, 'config_filepath': 'pretrained-512-pascal-plus', 'species_set' : set(['zebra']), 'classify': True},
        # {'label': 'COMBINED*', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True},
        # {'label': 'COMBINED* !0.1', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'conf_thresh': 0.1},
        # {'label': 'COMBINED* !0.5', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'conf_thresh': 0.5},
        # {'label': 'COMBINED* !0.9', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'conf_thresh': 0.9},
        # {'label': 'COMBINED* ~0.01', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'nms': True, 'nms_thresh': 0.01},
        # {'label': 'COMBINED* ~0.05', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'nms': True, 'nms_thresh': 0.05},
        # {'label': 'COMBINED* ~0.5', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'nms': True, 'nms_thresh': 0.5},
        # {'label': 'COMBINED* ~0.9', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'nms': True, 'nms_thresh': 0.9},

        # {'label': 'COMBINED 0.0', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.0},
        # {'label': 'COMBINED 0.1', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.1},
        # {'label': 'COMBINED 0.2', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.2},
        # {'label': 'COMBINED 0.3', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.3},
        # {'label': 'COMBINED 0.4', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.4},
        # {'label': 'COMBINED 0.5', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.5},
        # {'label': 'COMBINED 0.6', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.6},
        # {'label': 'COMBINED 0.7', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.7},
        # {'label': 'COMBINED 0.8', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.8},
        # {'label': 'COMBINED 0.9', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'p': 0.9},
        # {'label': 'COMBINED 1.0', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True, 'p': 1.0},
        # {'label': 'COMBINED MUL', 'algo': '_COMBINED', 'species_set' : set(['zebra']), 'classify': True},
    ]

    color_list = pt.distinct_colors(len(config_list), randomize=False)
    # color_list = pt.distinct_colors(len(config_list) // 2, randomize=False)
    # color_list = color_list + color_list

    ret_list = [
        localizer_precision_recall_algo_plot(ibs, color=color, min_overlap=min_overlap,
                                             x_limit=min_recall, **config)
        for color, config in zip(color_list, config_list)
    ]

    area_list = [ ret[0] for ret in ret_list ]
    conf_list = [ ret[1] for ret in ret_list ]
    index = np.argmax(area_list)
    best_label = config_list[index]['label']
    best_color = color_list[index]
    best_config = config_list[index]
    best_area = area_list[index]
    best_conf = conf_list[index]
    plt.title('Precision-Recall Curve (Best: %s, mAP = %0.02f)' % (best_label, best_area, ), y=1.13)

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
    args = (best_label, best_conf, )
    plt.title('P-R Confusion Matrix for Highest mAP\n(Algo: %s, OP = %0.02f)' % args, y=1.26)

    # Show best that is greater than the best_pr
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

        best_label = config_list[best_index]['label']
        best_color = color_list[index]
        best_config = config_list[best_index]

        axes_ = plt.subplot(133)
        axes_.set_aspect(1)
        gca_ = plt.gca()
        gca_.grid(False)
        correct_rate, _ = localizer_confusion_matrix_algo_plot(ibs, best_color, best_conf,
                                                               min_overlap=min_overlap,
                                                               fig_=fig_, axes_=axes_,
                                                               **best_config)
        axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
        axes_.set_ylabel('Ground-Truth')
        args = (min_recall, best_label, best_conf, )
        plt.title('P-R Confusion Matrix for Highest Precision with Recall >= %0.02f\n(Algo: %s, OP = %0.02f)' % args, y=1.26)

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


def classifier_precision_recall_algo(ibs, category_set, **kwargs):
    depc = ibs.depc_image
    test_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
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


def classifier_precision_recall_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    print('Processing Precision-Recall for: %r' % (label, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = classifier_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, re_list, pr_list, **kwargs)


def classifier_roc_algo_plot(ibs, **kwargs):
    label = kwargs['label']
    kwargs['invert'] = True
    print('Processing ROC for: %r' % (label, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = classifier_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, fpr_list, tpr_list, **kwargs)


def classifier_confusion_matrix_algo_plot(ibs, label, color, conf, category_set, **kwargs):
    print('Processing Confusion Matrix for: %r (Conf = %0.02f)' % (label, conf, ))
    depc = ibs.depc_image
    test_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
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
def classifier_precision_recall_algo_display(ibs, species_list, figsize=(16, 16), **kwargs):
    import matplotlib.pyplot as plt

    fig_ = plt.figure(figsize=figsize)

    category_set = set(species_list)

    kwargs['classifier_weight_filepath'] = 'coco_zebra'

    axes_ = plt.subplot(221)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall')
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    area, best_conf1, _ = classifier_precision_recall_algo_plot(ibs, label='V1', color='r', category_set=category_set, **kwargs)
    plt.title('Precision-Recall Curve (mAP = %0.02f)' % (area, ), y=1.10)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    axes_ = plt.subplot(222)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('False-Positive Rate')
    axes_.set_ylabel('True-Positive Rate')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    area, best_conf2, _ = classifier_roc_algo_plot(ibs, label='V1', color='r', category_set=category_set, **kwargs)
    plt.title('ROC Curve (mAP = %0.02f)' % (area, ), y=1.10)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    axes_ = plt.subplot(223)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = classifier_confusion_matrix_algo_plot(ibs, 'V1', 'r', conf=best_conf1, fig_=fig_, axes_=axes_, category_set=category_set, **kwargs)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    plt.title('P-R Confusion Matrix (OP = %0.02f)' % (best_conf1, ), y=1.12)

    axes_ = plt.subplot(224)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, _ = classifier_confusion_matrix_algo_plot(ibs, 'V1', 'r', conf=best_conf2, fig_=fig_, axes_=axes_, category_set=category_set, **kwargs)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%)' % (correct_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    plt.title('ROC Confusion Matrix (OP = %0.02f)' % (best_conf2, ), y=1.12)

    fig_filename = 'classifier-precision-recall-roc.png'
    fig_path = abspath(expanduser(join('~', 'Desktop', fig_filename)))
    plt.savefig(fig_path, bbox_inches='tight')


def labeler_tp_tn_fp_fn(ibs, category_list, samples=10000, **kwargs):
    # from ibeis.algo.detect.labeler.model import label_list as category_list

    def labeler_tp_tn_fp_fn_(zipped, conf, category):
        error_list = [0, 0, 0, 0]
        for index, (label, confidence) in enumerate(zipped):
            if label == category and conf <= confidence:
                error_list[0] += 1
            elif label != category and conf <= confidence:
                error_list[2] += 1
            elif label == category:
                error_list[3] += 1
            elif label != category:
                error_list[1] += 1
        return error_list

    depc = ibs.depc_annot
    test_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
    test_gid_set = list(test_gid_set)
    aids_list = ibs.get_image_aids(test_gid_set)
    aid_list = ut.flatten(aids_list)
    # Get annot species and yaws
    species_list = ibs.get_annot_species_texts(aid_list)
    yaw_list = ibs.get_annot_yaw_texts(aid_list)
    # Filter aids with species of interest and undefined yaws
    flag_list = [
        species in category_list and yaw is None
        for species, yaw in zip(species_list, yaw_list)
    ]
    flag_list = ut.not_list(flag_list)
    if False in flag_list:
        aid_list = ut.compress(yaw_list, )
        # Get new species and yaws
        yaw_list = ibs.get_annot_yaw_texts(aid_list)
        species_list = ibs.get_annot_species_texts(aid_list)
    # Make ground-truth
    label_list = [
        '%s:%s' % (species, yaw, ) if species in category_list else 'ignore'
        for species, yaw in zip(species_list, yaw_list)
    ]
    # Get predictions
    probability_dict_list = depc.get_property('labeler', aid_list, 'probs')
    conf_list = [ _ / float(samples) for _ in range(0, int(samples) + 1) ]

    label_dict = {}
    for category in category_list:
        print('\t%r' % (category, ))
        conf_dict = {}
        confidence_list = [
            probability_dict[category]
            for probability_dict in probability_dict_list
        ]
        zipped = list(zip(label_list, confidence_list))
        for conf in conf_list:
            conf_dict[conf] = labeler_tp_tn_fp_fn_(zipped, conf, category)
        label_dict[category] = conf_dict
    return label_dict


def labeler_precision_recall_algo(ibs, category_list, label_dict, **kwargs):

    global_conf_dict = {}
    for category in category_list:
        conf_dict = label_dict[category]
        for conf in conf_dict:
            new_list = conf_dict[conf]
            if conf not in global_conf_dict:
                global_conf_dict[conf] = new_list
            else:
                cur_list = global_conf_dict[conf]
                zipped_ = zip(cur_list, new_list)
                global_conf_dict[conf] = [cur + new for cur, new in zipped_]

    conf_list_ = [-1.0]
    pr_list = [1.0]
    re_list = [0.0]
    tpr_list = [0.0]
    fpr_list = [0.0]
    for conf in sorted(global_conf_dict.keys(), reverse=True):
        error_list = global_conf_dict[conf]
        tp, tn, fp, fn = error_list
        try:
            pr = tp / (tp + fp)
            re = tp / (tp + fn)
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
        except ZeroDivisionError:
            print('\tbad conf %0.05f - %d %d %d %d' % (conf, tp, tn, fp, fn, ))
            continue
        conf_list_.append(conf)
        pr_list.append(pr)
        re_list.append(re)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

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
    kwargs['invert'] = True
    print('Processing ROC for: %r (category_list = %r)' % (label, category_list, ))
    conf_list, pr_list, re_list, tpr_list, fpr_list = labeler_precision_recall_algo(ibs, **kwargs)
    return general_area_best_conf(conf_list, fpr_list, tpr_list, **kwargs)


def labeler_confusion_matrix_algo_plot(ibs, category_list, label, color, **kwargs):
    # from ibeis.algo.detect.labeler.model import label_list as category_list
    print('Processing Confusion Matrix for: %r' % (label, ))
    depc = ibs.depc_annot
    test_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
    test_gid_set = list(test_gid_set)
    aids_list = ibs.get_image_aids(test_gid_set)
    aid_list = ut.flatten(aids_list)
    species_list = ibs.get_annot_species_texts(aid_list)
    yaw_list = ibs.get_annot_yaw_texts(aid_list)
    label_list = [
        '%s:%s' % (species, yaw, ) if species in category_list else 'ignore'
        for species, yaw in zip(species_list, yaw_list)
    ]
    conf_list = depc.get_property('labeler', aid_list, 'score')
    species_list = depc.get_property('labeler', aid_list, 'species')
    yaw_list = depc.get_property('labeler', aid_list, 'viewpoint')
    prediction_list = [
        '%s:%s' % (species, yaw, ) if species in category_list else 'ignore'
        for species, yaw in zip(species_list, yaw_list)
    ]

    category_list = map(simple_code, category_list)
    label_list = map(simple_code, label_list)
    prediction_list = map(simple_code, prediction_list)
    category_mapping = { key: index for index, key in enumerate(category_list) }
    return general_confusion_matrix_algo(label_list, prediction_list, category_list,
                                                 category_mapping, conf_list=conf_list,
                                                 **kwargs)


@register_ibs_method
def labeler_precision_recall_algo_display(ibs, category_list=None, figsize=(16, 16),
                                          **kwargs):
    import matplotlib.pyplot as plt
    # from ibeis.algo.detect.labeler.model import label_list

    if category_list is None:
        test_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TEST_SET')))
        test_gid_set = list(test_gid_set)
        aids_list = ibs.get_image_aids(test_gid_set)
        aid_list = ut.flatten(aids_list)
        species_list = ibs.get_annot_species_texts(aid_list)
        category_list = sorted(list(set(species_list)))

    print('Compiling raw numbers...')
    label_dict = labeler_tp_tn_fp_fn(ibs, category_list, **kwargs)

    category_color_list = [
        (None          , 'b', 'All'),
        (':'           , 'y', 'Zebras'),
        ('zebra_plains', 'r', 'Plains Only'),
        ('zebra_grevys', 'g', 'Grevy\'s Only'),
        ('ignore'      , 'k', 'Ignore Only'),
    ]

    label_list = None  # TODO

    fig_ = plt.figure(figsize=figsize)  # NOQA

    axes_ = plt.subplot(221)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('Recall')
    axes_.set_ylabel('Precision')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    area_list = []
    conf_list = []
    for category, color, label_ in category_color_list:
        category_list = [ label for label in label_list if category is None or category in label ]
        area, conf, _ = labeler_precision_recall_algo_plot(ibs, category_list=category_list, label=label_, color=color, label_dict=label_dict)
        area_list.append(area)
        conf_list.append(conf)
    best_area = area_list[0]
    best_conf = conf_list[0]
    plt.title('Precision-Recall Curve (Algo: All, mAP = %0.02f)' % (best_area, ), y=1.19)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    axes_ = plt.subplot(222)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_xlabel('False-Positive Rate')
    axes_.set_ylabel('True-Positive Rate')
    axes_.set_xlim([0.0, 1.01])
    axes_.set_ylim([0.0, 1.01])
    area_list = []
    conf_list = []
    for category, color, label_ in category_color_list:
        category_list = [ label for label in label_list if category is None or category in label ]
        area, conf, _ = labeler_roc_algo_plot(ibs, category_list=category_list, label=label_, color=color, label_dict=label_dict)
        area_list.append(area)
        conf_list.append(conf)
    best_area = area_list[0]
    plt.title('ROC Curve (Algo: All, mAP = %0.02f)' % (best_area, ), y=1.19)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    fuzzy_dict = {}
    for index1, label1 in enumerate(label_list):
        if label1 == 'ignore':
            fuzzy_list = []
        else:
            species, viewpoint = label1.strip().split(':')
            fuzzy_list = []
            for index2, label2 in enumerate(label_list):
                if species in label2:
                    fuzzy_list.append(index2)
        fuzzy_dict[index1] = set(fuzzy_list)
    axes_ = plt.subplot(223)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, fuzzy_rate = labeler_confusion_matrix_algo_plot(ibs, 'V1', 'r', category_list=category_list, fig_=fig_, axes_=axes_, fuzzy_dict=fuzzy_dict, conf=None)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%, Species = %0.02f%%)' % (correct_rate * 100.0, fuzzy_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    plt.title('P-R Confusion Matrix', y=1.15)

    axes_ = plt.subplot(224)
    axes_.set_aspect(1)
    gca_ = plt.gca()
    gca_.grid(False)
    correct_rate, fuzzy_rate = labeler_confusion_matrix_algo_plot(ibs, 'V1', 'r', category_list=category_list, fig_=fig_, axes_=axes_, fuzzy_dict=fuzzy_dict, conf=best_conf)
    axes_.set_xlabel('Predicted (Correct = %0.02f%%, Species = %0.02f%%)' % (correct_rate * 100.0, fuzzy_rate * 100.0, ))
    axes_.set_ylabel('Ground-Truth')
    plt.title('P-R Confusion Matrix (Algo: All, OP = %0.02f)' % (best_conf, ), y=1.15)

    fig_filename = 'labeler-precision-recall-roc.png'
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
                'xtl'        : bbox[0] / width,
                'ytl'        : bbox[1] / height,
                'xbr'        : (bbox[0] + bbox[2]) / width,
                'ybr'        : (bbox[1] + bbox[3]) / height,
                'width'      : bbox[2] / width,
                'height'     : bbox[3] / height,
                'species'    : ibs.get_annot_species_texts(aid),
                'viewpoint'  : ibs.get_annot_yaw_texts(aid),
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
                'xtl'        : bbox[0] / width,
                'ytl'        : bbox[1] / height,
                'width'      : bbox[2] / width,
                'height'     : bbox[3] / height,
                'theta'      : theta,  # round(theta, 4),
                'confidence' : conf,   # round(conf, 4),
                'species'    : species_,
                'viewpoint'  : viewpoint,
            }
            for bbox, theta, species_, viewpoint, conf in zip(*zipped[0][1:])
        ]
        for zipped, (width, height) in zip(zipped_list, size_list)
    ]

    pred_dict = {
        uuid_ : result_list
        for uuid_, result_list in zip(uuid_list, results_list)
    }
    # print(pred_dict)
    return pred_dict


def detector_precision_recall_algo(ibs, samples=1000, force_serial=True, **kwargs):
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
    pr_re_gen = ut.generate(detector_precision_recall_algo_worker, arg_iter,
                            nTasks=len(conf_list), ordered=True,
                            chunksize=50, force_serial=force_serial)

    conf_list_ = [-1.0]
    pr_list = [1.0]
    re_list = [0.0]
    for conf, pr, re in pr_re_gen:
        conf_list_.append(conf)
        pr_list.append(pr)
        re_list.append(re)

    print('...complete')
    return conf_list_, pr_list, re_list


def detector_precision_recall_algo_worker(tup):
    conf, uuid_list, gt_dict, pred_dict, kwargs = tup
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
        detector_precision_recall_algo_plot(ibs, label=label, color=color, x_limit=0.5, **kwargs_)
        for label, color, kwargs_ in zip(label_list, color_list, kwargs_list)
    ]

    area_list = [ ret[0] for ret in ret_list ]
    conf_list = [ ret[1] for ret in ret_list ]
    index = np.argmax(area_list)
    best_label = label_list[index]
    best_kwargs = kwargs_list[index]
    best_area = area_list[index]
    best_conf = conf_list[index]
    plt.title('Precision-Recall Curve (Best: %s, mAP = %0.02f)' % (best_label, best_area, ), y=1.20)
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


@register_ibs_method
def classifier_train_svm(ibs, species_list):
    from sklearn import svm
    # import pickle

    # Load data
    print('Loading pre-trained features for images')
    depc = ibs.depc_image
    train_gid_set = general_get_imageset_gids(ibs, 'TRAIN_SET')
    config = {
        'algo': 'vgg16',
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

    print('Train SVM model using features and target labels')
    # Train new model using data and labels
    model = svm.SVC()
    model.fit(data_list, label_list)

    # pickle.dumps(model)
    # clf2 = pickle.loads(s)
    # clf2.predict(X[0:1])


@register_ibs_method
def classifier_train(ibs, species_list):
    from ibeis_cnn.ingest_ibeis import get_cnn_classifier_binary_training_images
    from ibeis_cnn.process import numpy_processed_directory2
    from ibeis_cnn.models.classifier import train_classifier
    from ibeis_cnn.utils import save_model
    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_cnn_classifier_binary_training_images(ibs, species_list, dest_path=data_path)
    id_file, X_file, y_file = numpy_processed_directory2(extracted_path)
    output_path = join(ibs.get_cachedir(), 'training', 'classifier')
    model_path = train_classifier(output_path, X_file, y_file)
    # Add the species_list to the model
    model_state = ut.load_cPkl(model_path)
    assert 'species_list' not in model_state
    model_state['species_list'] = species_list
    save_model(model_state, model_path)
    # Return model path
    return model_path


@register_ibs_method
def localizer_train(ibs, **kwargs):
    from pydarknet import Darknet_YOLO_Detector
    data_path = ibs.export_to_xml(**kwargs)
    output_path = join(ibs.get_cachedir(), 'training', 'localizer')
    ut.ensuredir(output_path)
    dark = Darknet_YOLO_Detector()
    model_path = dark.train(data_path, output_path)
    del dark
    return model_path


@register_ibs_method
def labeler_train(ibs):
    from ibeis_cnn.ingest_ibeis import get_cnn_labeler_training_images
    from ibeis_cnn.process import numpy_processed_directory2
    from ibeis_cnn.models.labeler import train_labeler
    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_cnn_labeler_training_images(ibs, data_path)
    id_file, X_file, y_file = numpy_processed_directory2(extracted_path)
    output_path = join(ibs.get_cachedir(), 'training', 'labeler')
    model_path = train_labeler(output_path, X_file, y_file)
    return model_path


@register_ibs_method
def qualifier_train(ibs):
    from ibeis_cnn.ingest_ibeis import get_cnn_qualifier_training_images
    from ibeis.algo.detect.qualifier.qualifier import train_qualifier
    data_path = join(ibs.get_cachedir(), 'extracted')
    get_cnn_qualifier_training_images(ibs, data_path)
    output_path = join(ibs.get_cachedir(), 'training', 'qualifier')
    model_path = train_qualifier(output_path, source_path=data_path)
    return model_path


@register_ibs_method
def detector_train(ibs):
    results = ibs.localizer_train()
    localizer_weight_path, localizer_config_path, localizer_class_path = results
    classifier_model_path = ibs.classifier_train()
    labeler_model_path = ibs.labeler_train()
    output_path = join(ibs.get_cachedir(), 'training', 'detector')
    ut.ensuredir(output_path)
    ut.copy(localizer_weight_path, join(output_path, 'localizer.weights'))
    ut.copy(localizer_config_path, join(output_path, 'localizer.config'))
    ut.copy(localizer_class_path,  join(output_path, 'localizer.classes'))
    ut.copy(classifier_model_path, join(output_path, 'classifier.npy'))
    ut.copy(labeler_model_path,    join(output_path, 'labeler.npy'))


@register_ibs_method
def background_train(ibs):
    from ibeis_cnn.ingest_ibeis import get_background_training_patches2
    from ibeis_cnn.process import numpy_processed_directory2
    from ibeis_cnn.models.background import train_background
    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_background_training_patches2(ibs, data_path,
                                                      patch_size=50,
                                                      global_limit=500000)
    id_file, X_file, y_file = numpy_processed_directory2(extracted_path)
    output_path = join(ibs.get_cachedir(), 'training', 'background')
    model_path = train_background(output_path, X_file, y_file)
    return model_path


def _resize(image, t_width=None, t_height=None, verbose=True):
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
