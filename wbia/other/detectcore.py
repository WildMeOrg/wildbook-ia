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
from six.moves import zip
from os.path import exists, expanduser, join, abspath
import numpy as np
import vtool as vt
import utool as ut
import cv2
import wbia.constants as const
from wbia.control import controller_inject
from wbia.other.detectfuncs import (
    general_parse_gt,
    general_get_imageset_gids,
    localizer_parse_pred,
    _resize,
    general_overlap,
)
import random

# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[other.detectcore]')


CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)


def nms(dets, scores, thresh, use_cpu=True):
    # Interface into Faster R-CNN's Python native NMS algorithm by Girshick et al.
    from wbia.algo.detect.nms.py_cpu_nms import py_cpu_nms

    return py_cpu_nms(dets, scores, thresh)


@register_ibs_method
def export_to_pascal(ibs, *args, **kwargs):
    """Alias for export_to_xml"""
    return ibs.export_to_xml(*args, **kwargs)


@register_ibs_method
def export_to_xml(
    ibs,
    species_list,
    species_mapping=None,
    offset='auto',
    enforce_viewpoint=False,
    target_size=900,
    purge=False,
    use_maximum_linear_dimension=True,
    use_existing_train_test=True,
    include_parts=False,
    gid_list=None,
    output_path=None,
    allow_empty_images=False,
    min_annot_size=5,
    **kwargs,
):
    """Create training XML for training models."""
    import random
    from datetime import date
    from wbia.detecttools.pypascalmarkup import PascalVOC_Markup_Annotation

    print('Received species_mapping = %r' % (species_mapping,))

    if species_list is None:
        species_list = sorted(set(species_mapping.values()))

    print('Using species_list = %r' % (species_list,))

    def _add_annotation(
        annotation,
        bbox,
        theta,
        species_name,
        viewpoint,
        interest,
        decrease,
        width,
        height,
        part_type=None,
    ):
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

        w_ = xmax - xmin
        h_ = ymax - ymin
        if w_ < min_annot_size:
            return
        if h_ < min_annot_size:
            return

        if viewpoint != -1 and viewpoint is not None:
            info['pose'] = viewpoint

        if interest is not None:
            info['interest'] = '1' if interest else '0'

        if part_type is not None:
            species_name = '%s+%s' % (species_name, part_type,)

        area = w_ * h_
        print('\t\tAdding %r with area %0.04f pixels^2' % (species_name, area,))

        annotation.add_object(species_name, (xmax, xmin, ymax, ymin), **info)

    current_year = int(date.today().year)
    information = {'database_name': ibs.get_dbname()}
    import datetime

    now = datetime.datetime.now()
    folder = 'VOC%d' % (now.year,)

    if output_path is None:
        output_path = ibs.get_cachedir()

    datadir = join(output_path, 'VOCdevkit', folder)
    imagedir = join(datadir, 'JPEGImages')
    annotdir = join(datadir, 'Annotations')
    setsdir = join(datadir, 'ImageSets')
    mainsetsdir = join(setsdir, 'Main')

    if purge:
        ut.delete(datadir)

    ut.ensuredir(datadir)
    ut.ensuredir(imagedir)
    ut.ensuredir(annotdir)
    ut.ensuredir(setsdir)
    ut.ensuredir(mainsetsdir)

    # Get all gids and process them
    if gid_list is None:
        gid_list = sorted(ibs.get_valid_gids())

    sets_dict = {
        'test': [],
        'train': [],
        'trainval': [],
        'val': [],
    }
    index = 1 if offset == 'auto' else offset

    # Make a preliminary train / test split as imagesets or use the existing ones
    if not use_existing_train_test:
        ibs.imageset_train_test_split(**kwargs)

    train_gid_set = set(general_get_imageset_gids(ibs, 'TRAIN_SET', **kwargs))
    test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET', **kwargs))

    print('Exporting %d images' % (len(gid_list),))
    for gid in gid_list:
        aid_list = ibs.get_image_aids(gid)
        image_uri = ibs.get_image_uris(gid)
        image_path = ibs.get_image_paths(gid)
        if len(aid_list) > 0 or allow_empty_images:
            fulldir = image_path.split('/')
            filename = fulldir.pop()
            extension = filename.split('.')[-1]  # NOQA
            out_name = '%d_%06d' % (current_year, index,)
            out_img = '%s.jpg' % (out_name,)

            _image = ibs.get_images(gid)
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

            dst_img = join(imagedir, out_img)
            _image = vt.resize(_image, (width, height))
            vt.imwrite(dst_img, _image)

            annotation = PascalVOC_Markup_Annotation(
                dst_img, folder, out_img, source=image_uri, **information
            )
            bbox_list = ibs.get_annot_bboxes(aid_list)
            theta_list = ibs.get_annot_thetas(aid_list)
            species_name_list = ibs.get_annot_species_texts(aid_list)
            viewpoint_list = ibs.get_annot_viewpoints(aid_list)
            interest_list = ibs.get_annot_interest(aid_list)

            part_rowids_list = ibs.get_annot_part_rowids(aid_list)
            zipped = zip(
                bbox_list,
                theta_list,
                species_name_list,
                viewpoint_list,
                interest_list,
                part_rowids_list,
            )
            for (
                bbox,
                theta,
                species_name,
                viewpoint,
                interest,
                part_rowid_list,
            ) in zipped:
                if species_mapping is not None:
                    species_name = species_mapping.get(species_name, species_name)

                if species_name is not None and species_name not in species_list:
                    continue

                _add_annotation(
                    annotation,
                    bbox,
                    theta,
                    species_name,
                    viewpoint,
                    interest,
                    decrease,
                    width,
                    height,
                )

                if include_parts and len(part_rowid_list) > 0:
                    part_bbox_list = ibs.get_part_bboxes(part_rowid_list)
                    part_theta_list = ibs.get_part_thetas(part_rowid_list)
                    part_type_list = ibs.get_part_types(part_rowid_list)
                    part_zipped = zip(part_bbox_list, part_theta_list, part_type_list)
                    for part_bbox, part_theta, part_type in part_zipped:
                        part_viewpoint = viewpoint
                        part_interest = None
                        _add_annotation(
                            annotation,
                            part_bbox,
                            part_theta,
                            species_name,
                            part_viewpoint,
                            part_interest,
                            decrease,
                            width,
                            height,
                            part_type=part_type,
                        )

            out_filename = '%s.xml' % (out_name,)
            dst_annot = join(annotdir, out_filename)

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
                raise AssertionError(
                    'All gids must be either in the TRAIN_SET or TEST_SET imagesets'
                )

            # Write XML
            print('Copying:\n%r\n%r\n%r\n\n' % (image_path, dst_img, (width, height),))
            xml_data = open(dst_annot, 'w')
            xml_data.write(annotation.xml())
            xml_data.close()
            while exists(dst_annot):
                index += 1
                if offset != 'auto':
                    break
                out_filename = '%d_%06d.xml' % (current_year, index,)
                dst_annot = join(annotdir, out_filename)
        else:
            print('Skipping:\n%r\n\n' % (image_path,))

    for key in sets_dict.keys():
        manifest_filename = '%s.txt' % (key,)
        manifest_filepath = join(mainsetsdir, manifest_filename)
        with open(manifest_filepath, 'w') as file_:
            sets_dict[key].append('')
            content = sets_dict[key]
            content = '\n'.join(content)
            file_.write(content)

    print('...completed')
    return datadir


@register_ibs_method
def export_to_coco(
    ibs,
    species_list,
    species_mapping={},
    viewpoint_mapping={},
    target_size=2400,
    use_maximum_linear_dimension=True,
    use_existing_train_test=True,
    include_parts=False,
    gid_list=None,
    include_reviews=True,
    require_image_reviewed=False,
    require_named=False,
    output_images=True,
    **kwargs,
):
    """Create training COCO dataset for training models."""
    from datetime import date
    import datetime
    import random
    import json

    print('Received species_mapping = %r' % (species_mapping,))
    print('Received viewpoint_mapping = %r' % (viewpoint_mapping,))

    if species_list is None:
        species_list = sorted(set(species_mapping.values()))

    print('Using species_list = %r' % (species_list,))

    current_year = int(date.today().year)
    datadir = abspath(join(ibs.get_cachedir(), 'coco'))
    annotdir = join(datadir, 'annotations')
    imagedir = join(datadir, 'images')
    image_dir_dict = {
        'train': join(imagedir, 'train%s' % (current_year,)),
        'val': join(imagedir, 'val%s' % (current_year,)),
        'test': join(imagedir, 'test%s' % (current_year,)),
    }

    ut.delete(datadir)
    ut.ensuredir(datadir)
    ut.ensuredir(annotdir)
    ut.ensuredir(imagedir)
    for dataset in image_dir_dict:
        ut.ensuredir(image_dir_dict[dataset])

    info = {
        'description': 'Wild Me %s Dataset' % (ibs.dbname,),
        # 'url'          : 'http://www.greatgrevysrally.com',
        'url': 'http://www.wildme.org',
        'version': '1.0',
        'year': current_year,
        'contributor': 'Wild Me, Jason Parham <parham@wildme.org>',
        'date_created': datetime.datetime.utcnow().isoformat(' '),
        'name': ibs.get_db_name(),
        'uuid': str(ibs.get_db_init_uuid()),
    }

    licenses = [
        {
            'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/',
            'id': 3,
            'name': 'Attribution-NonCommercial-NoDerivs License',
        },
    ]

    assert len(species_list) == len(
        set(species_list)
    ), 'Cannot have duplicate species in species_list'
    category_dict = {}
    categories = []
    for index, species in enumerate(sorted(species_list)):

        species = species_mapping.get(species, species)

        categories.append({'id': index, 'name': species, 'supercategory': 'animal'})
        category_dict[species] = index

    def _add_annotation_or_part(
        image_index,
        annot_index,
        annot_uuid,
        bbox,
        theta,
        species_name,
        viewpoint,
        interest,
        annot_name,
        decrease,
        width,
        height,
        part_index=None,
        part_uuid=None,
    ):
        is_part = part_index is not None

        R = vt.rotation_around_bbox_mat3x3(theta, bbox)
        verts = vt.verts_from_bbox(bbox, close=True)
        xyz_pts = vt.add_homogenous_coordinate(np.array(verts).T)
        trans_pts = vt.remove_homogenous_coordinate(R.dot(xyz_pts))
        new_verts = np.round(trans_pts).astype(np.int).T.tolist()

        x_points = [int(np.around(pt[0] * decrease)) for pt in new_verts]
        y_points = [int(np.around(pt[1] * decrease)) for pt in new_verts]
        segmentation = ut.flatten(list(zip(x_points, y_points)))

        xmin = max(min(x_points), 0)
        ymin = max(min(y_points), 0)
        xmax = min(max(x_points), width - 1)
        ymax = min(max(y_points), height - 1)

        w = xmax - xmin
        h = ymax - ymin
        area = w * h

        xtl_, ytl_, w_, h_ = bbox
        xtl_ *= decrease
        ytl_ *= decrease
        w_ *= decrease
        h_ *= decrease

        annot_part = {
            'bbox': [xtl_, ytl_, w_, h_],
            'theta': theta,
            'viewpoint': viewpoint,
            'segmentation': [segmentation],
            'segmentation_bbox': [xmin, ymin, w, h],
            'area': area,
            'iscrowd': 0,
            'id': part_index if is_part else annot_index,
            'image_id': image_index,
            'category_id': category_dict[species_name],
            'uuid': str(part_uuid if is_part else annot_uuid),
            # 'individual_ids'    : individuals,
        }
        if is_part:
            annot_part['annot_id'] = annot_index
        else:
            annot_part['isinterest'] = int(interest)
            annot_part['name'] = annot_name

        return annot_part, area

    output_dict = {}
    for dataset in ['train', 'val', 'test']:
        output_dict[dataset] = {
            'info': info,
            'licenses': licenses,
            'categories': categories,
            'images': [],
            'annotations': [],
            'parts': [],
        }

    # Get all gids and process them
    if gid_list is None:
        if require_named:
            aid_list = ibs.get_valid_aids()
            species_list_ = ibs.get_annot_species(aid_list)
            flag_list = [
                species_mapping.get(species_, species_) in species_list
                for species_ in species_list_
            ]
            aid_list = ut.compress(aid_list, flag_list)
            nid_list = ibs.get_annot_nids(aid_list)
            flag_list = [nid >= 0 for nid in nid_list]
            aid_list = ut.compress(aid_list, flag_list)
            gid_list = list(set(ibs.get_annot_gids(aid_list)))
        else:
            gid_list = ibs.get_valid_gids()

        if require_image_reviewed:
            image_reviewed_list = ibs.get_image_reviewed(gid_list)
            gid_list = ut.compress(gid_list, image_reviewed_list)

        gid_list = sorted(list(set(gid_list)))

    # Make a preliminary train / test split as imagesets or use the existing ones
    if not use_existing_train_test:
        ibs.imageset_train_test_split(**kwargs)

    train_gid_set = set(general_get_imageset_gids(ibs, 'TRAIN_SET', **kwargs))
    test_gid_set = set(general_get_imageset_gids(ibs, 'TEST_SET', **kwargs))

    image_index = 1
    annot_index = 1
    part_index = 1

    aid_dict = {}

    print('Exporting %d images' % (len(gid_list),))
    for gid in gid_list:

        if gid in test_gid_set:
            dataset = 'test'
        elif gid in train_gid_set:
            state = random.uniform(0.0, 1.0)
            if state <= 0.75:
                dataset = 'train'
            else:
                dataset = 'val'
        else:
            # raise AssertionError('All gids must be either in the TRAIN_SET or TEST_SET imagesets')
            print('GID = %r was not in the TRAIN_SET or TEST_SET' % (gid,))
            dataset = 'test'

        width, height = ibs.get_image_sizes(gid)
        if target_size is None:
            decrease = 1.0
        else:
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

        image_path = ibs.get_image_paths(gid)
        image_filename = '%012d.jpg' % (image_index,)
        image_filepath = join(image_dir_dict[dataset], image_filename)

        if output_images:
            _image = ibs.get_images(gid)
            _image = vt.resize(_image, (width, height))
            vt.imwrite(image_filepath, _image)

        output_dict[dataset]['images'].append(
            {
                'license': 3,
                'file_name': image_filename,
                # 'file_name'     : basename(ibs.get_image_uris_original(gid)),
                'coco_url': None,
                'height': height,
                'width': width,
                'date_captured': ibs.get_image_datetime_str(gid).replace('/', '-'),
                'flickr_url': None,
                'id': image_index,
                'uuid': str(ibs.get_image_uuids(gid)),
            }
        )

        print('Copying:\n%r\n%r\n%r\n\n' % (image_path, image_filepath, (width, height),))

        aid_list = ibs.get_image_aids(gid)
        bbox_list = ibs.get_annot_bboxes(aid_list)
        theta_list = ibs.get_annot_thetas(aid_list)
        species_name_list = ibs.get_annot_species_texts(aid_list)
        viewpoint_list = ibs.get_annot_viewpoints(aid_list)
        interest_list = ibs.get_annot_interest(aid_list)
        annot_uuid_list = ibs.get_annot_uuids(aid_list)
        annot_name_list = ibs.get_annot_name_texts(aid_list)
        part_rowids_list = ibs.get_annot_part_rowids(aid_list)
        nid_list = ibs.get_annot_nids(aid_list)

        zipped = zip(
            aid_list,
            bbox_list,
            theta_list,
            species_name_list,
            viewpoint_list,
            interest_list,
            annot_uuid_list,
            annot_name_list,
            part_rowids_list,
            nid_list,
        )
        for (
            aid,
            bbox,
            theta,
            species_name,
            viewpoint,
            interest,
            annot_uuid,
            annot_name,
            part_rowid_list,
            nid,
        ) in zipped:
            species_name = species_mapping.get(species_name, species_name)

            if species_name is None:
                continue

            if species_name not in species_list:
                continue

            if require_named and nid < 0:
                continue

            viewpoint = viewpoint_mapping.get(species_name, {}).get(viewpoint, viewpoint)

            # if viewpoint is None:
            #     continue

            # Transformation matrix
            annot, area = _add_annotation_or_part(
                image_index,
                annot_index,
                annot_uuid,
                bbox,
                theta,
                species_name,
                viewpoint,
                interest,
                annot_name,
                decrease,
                width,
                height,
            )
            print('\t\tAdding annot %r with area %0.04f pixels^2' % (species_name, area,))

            if include_reviews:
                # individuals = ibs.get_name_aids(ibs.get_annot_nids(aid))
                reviews = ibs.get_review_rowids_from_single([aid])[0]
                user_list = ibs.get_review_identity(reviews)
                aid_tuple_list = ibs.get_review_aid_tuple(reviews)
                decision_list = ibs.get_review_decision_str(reviews)

                ids = []
                decisions = []
                zipped = zip(user_list, aid_tuple_list, decision_list)
                for user, aid_tuple, decision in zipped:
                    if 'user:web' not in user:
                        continue
                    match = list(set(aid_tuple) - set([aid]))
                    assert len(match) == 1
                    ids.append(match[0])
                    decisions.append(decision.lower())
                annot['review_ids'] = list(zip(ids, decisions))

            output_dict[dataset]['annotations'].append(annot)

            if include_parts and len(part_rowid_list) > 0:

                part_uuid_list = ibs.get_part_uuids(part_rowid_list)
                part_bbox_list = ibs.get_part_bboxes(part_rowid_list)
                part_theta_list = ibs.get_part_thetas(part_rowid_list)
                part_type_list = ibs.get_part_types(part_rowid_list)

                part_zipped = zip(
                    part_uuid_list, part_bbox_list, part_theta_list, part_type_list
                )
                for part_uuid, part_bbox, part_theta, part_type in part_zipped:
                    part_species_name = '%s+%s' % (species_name, part_type,)

                    part_species_name = species_mapping.get(
                        part_species_name, part_species_name
                    )

                    if part_species_name is None:
                        continue

                    if part_species_name not in species_list:
                        continue

                    part, area = _add_annotation_or_part(
                        image_index,
                        annot_index,
                        annot_uuid,
                        part_bbox,
                        part_theta,
                        part_species_name,
                        viewpoint,
                        interest,
                        annot_name,
                        decrease,
                        width,
                        height,
                        part_index=part_index,
                        part_uuid=part_uuid,
                    )
                    print(
                        '\t\tAdding part %r with area %0.04f pixels^2'
                        % (part_species_name, area,)
                    )
                    output_dict[dataset]['parts'].append(part)

                part_index += 1

            aid_dict[aid] = annot_index
            annot_index += 1

        image_index += 1

    for dataset in output_dict:
        annots = output_dict[dataset]['annotations']
        for index in range(len(annots)):
            annot = annots[index]

            # Map internal aids to external annot index
            # individual_ids = annot['individual_ids']
            # individual_ids_ = []
            # for individual_id in individual_ids:
            #     if individual_id not in aid_dict:
            #         continue
            #     individual_id_ = aid_dict[individual_id]
            #     individual_ids_.append(individual_id_)
            # annot['individual_ids'] = individual_ids_

            # Map reviews
            if include_reviews:
                review_ids = annot['review_ids']
                review_ids_ = []
                for review in review_ids:
                    review_id, review_decision = review
                    if review_id not in aid_dict:
                        continue
                    review_id_ = aid_dict[review_id]
                    review_ = (
                        review_id_,
                        review_decision,
                    )
                    review_ids_.append(review_)
                annot['review_ids'] = review_ids_

            # Store
            output_dict[dataset]['annotations'][index] = annot

    for dataset in output_dict:
        json_filename = 'instances_%s%s.json' % (dataset, current_year,)
        json_filepath = join(annotdir, json_filename)

        with open(json_filepath, 'w') as json_file:
            json.dump(output_dict[dataset], json_file)

    print('...completed')
    return datadir


@register_ibs_method
def imageset_train_test_split(
    ibs, train_split=0.8, is_tile=False, gid_list=None, **kwargs
):
    from random import shuffle

    if gid_list is None:
        gid_list = ibs.get_valid_gids(is_tile=is_tile)

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
        split_index = total * (1.0 - train_split) + 1e-9  # weird
        if split_index < 1.0:
            split_index = total / 2
        else:
            split_index = np.around(split_index)
        split_index = int(split_index)
        args = (
            distro,
            total,
            split_index,
        )
        print('\tnum aids distro: %d - total: %d - split_index: %d' % args)
        train_list = gid_list_[split_index:]
        test_list = gid_list_[:split_index]
        args = (
            len(test_list),
            len(train_list),
            len(train_list) / total,
        )
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
        total = '%s' % (total,) if total < threshold else '%s+' % (total,)
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
        print('Species viewpoint distribution: %r' % (species,))
        viewpoint_dict = species_dict[species]
        total = 0
        for viewpoint in const.VIEWTEXT_TO_VIEWPOINT_RADIANS:
            count = viewpoint_dict.get(viewpoint, 0)
            print('{:>15}: {:>5}'.format(viewpoint, count))
            total += count
        print('TOTAL: %d\n' % (total,))

    # visualize_distributions(distro_dict, threshold=threshold)


def visualize_distributions(distro_dict, threshold=10):
    import matplotlib.pyplot as plt

    key_list = sorted(distro_dict.keys())
    threshold_str = '%d+' % (threshold,)
    label_list = [threshold_str if key == threshold else str(key) for key in key_list]
    size_list = [distro_dict[key] for key in key_list]
    color_list = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = [0.0] + [0.0] * (len(size_list) - 1)

    plt.pie(
        size_list,
        explode=explode,
        labels=label_list,
        colors=color_list,
        autopct='%1.1f%%',
        shadow=True,
        startangle=90,
    )
    plt.axis('equal')
    plt.show()


@register_ibs_method
def visualize_pascal_voc_dataset(
    ibs, dataset_path, num_examples=30, randomize=False, write=True, write_path=None
):
    r"""Visualize the PASCAL VOC dataset.

    Args:
        ibs (IBEISController):
        dataset_path (str): the dataset path in the PASCAL VOC format
        num_examples (int, optional): the number of examples to draw
        randomize (bool, optional): if to randomize the visualization
        write (bool, optional): if to display or write the files

    CommandLine:
        python -m wbia.other.detectcore --test-visualize_pascal_voc_dataset

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.detectcore import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> dataset_path = '/Users/jason.parham/Downloads/wilddog_data/data/VOCdevkit/VOC2020/'
        >>> # dataset_path = '/Users/jason.parham/Downloads/LearningData/'
        >>> # dataset_path = '/Users/jason.parham/Downloads/VOCdevkit/VOC2018/'
        >>> ibs.visualize_pascal_voc_dataset(dataset_path, randomize=True)
    """
    from wbia.detecttools.wbiadata import IBEIS_Data
    import random

    num_examples = int(num_examples)
    assert num_examples > 0

    dataset = IBEIS_Data(dataset_path)
    dataset.print_distribution()

    image_list = sorted(dataset.images)

    num_examples = min(num_examples, len(image_list))

    if randomize:
        random.shuffle(image_list)

    if write_path is None:
        write_path = abspath(expanduser(join('~', 'Desktop')))

    for image in image_list[:num_examples]:
        if write:
            write_filepath = join(write_path, image.filename)
            image = image.show(display=False)
            cv2.imwrite(write_filepath, image)
        else:
            image.show()


@register_ibs_method
def classifier_visualize_training_localizations(
    ibs,
    classifier_weight_filepath,
    species_list=['zebra'],
    scheme=2,
    output_path=None,
    values=None,
    **kwargs,
):
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
            chip = image[ytl:ybr, xtl:xbr, :]
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
            'algo': '_COMBINED',
            'species_set': set(species_list),
            'features': True,
            'feature2_algo': 'resnet',
            'classify': True,
            'classifier_algo': 'svm',
            'classifier_weight_filepath': classifier_weight_filepath,
            'nms': True,
            'nms_thresh': 0.50,
            # 'thresh'       : True,
            # 'index_thresh' : 0.25,
        }

        print('\tGather Ground-Truth')
        gt_dict = general_parse_gt(ibs, test_gid_list=train_gid_list, **config)

        print('\tGather Predictions')
        pred_dict = localizer_parse_pred(ibs, test_gid_list=train_gid_list, **config)

        print('Mine proposals')
        reviewed_gid_dict = {}
        values = _bootstrap_mine(
            ibs, gt_dict, pred_dict, scheme, reviewed_gid_dict, **kwargs
        )

    mined_gid_list, mined_gt_list, mined_pos_list, mined_neg_list = values

    print('Prepare images')
    # Get images and a dictionary based on their gids
    image_list = ibs.get_images(mined_gid_list)
    image_dict = {gid: image for gid, image in zip(mined_gid_list, image_list)}

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

    print('Write images to %r' % (output_path,))
    # Write images to disk
    for gid in image_dict:
        output_filename = 'localizations_gid_%d.png' % (gid,)
        output_filepath = join(output_path, output_filename)
        cv2.imwrite(output_filepath, image_dict[gid])


@register_ibs_method
def redownload_detection_models(ibs):
    r"""Re-download detection models.

    Args:
        ibs (IBEISController):

    CommandLine:
        python -c "from wbia.algo.detect import grabmodels; grabmodels.redownload_models()"
        python -c "import utool, wbia.algo; utool.view_directory(wbia.algo.detect.grabmodels._expand_modeldir())"

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.detectcore import *  # NOQA
        >>> import wbia  # NOQA
        >>> ibs = wbia.opendb('testdb1')
        >>> result = redownload_detection_models(ibs)
        >>> print(result)
    """
    print('[other.detectcore] redownload_detection_models')
    from wbia.algo.detect import grabmodels

    modeldir = ibs.get_detect_modeldir()
    grabmodels.redownload_models(modeldir=modeldir)


@register_ibs_method
def view_model_dir(ibs):
    print('[other.detectcore] redownload_detection_models')
    modeldir = ibs.get_detect_modeldir()
    ut.view_directory(modeldir)
    # grabmodels.redownload_models(modeldir=modeldir)


def _bootstrap_mine(
    ibs,
    gt_dict,
    pred_dict,
    scheme,
    reviewed_gid_dict,
    min_overlap=0.75,
    max_overlap=0.25,
):
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
        gt_list_ = [gt_list[picked_index] for picked_index in picked_index_list]

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

    args = (
        len(mined_pos_list),
        len(mined_neg_list),
        len(mined_gid_list),
    )
    print('Mined %d positive, %d negative from %d images' % args)

    return mined_gid_list, mined_gt_list, mined_pos_list, mined_neg_list


@register_ibs_method
def visualize_ground_truth(ibs, config, **kwargs):
    visualize_bounding_boxes(ibs, config, 'ground_truth', **kwargs)


@register_ibs_method
def visualize_predictions(ibs, config, **kwargs):
    visualize_bounding_boxes(ibs, config, 'prediction', **kwargs)


def visualize_bounding_boxes(
    ibs,
    config,
    version,
    gid_list=None,
    randomize=False,
    num_images=10,
    t_width=500,
    output_path=None,
):
    if gid_list is None:
        gid_list = general_get_imageset_gids(ibs, 'TEST_SET', **config)
    else:
        num_images = None

    if randomize:
        random.shuffle(gid_list)

    if num_images not in [-1, None]:
        num_images = min(num_images, len(gid_list))
        gid_list = gid_list[:num_images]

    uuid_list = ibs.get_image_uuids(gid_list)

    assert version is not None
    version = version.lower()
    if version == 'prediction':
        print('\tGather Predictions')
        val_dict = localizer_parse_pred(ibs, test_gid_list=gid_list, **config)
    elif version == 'ground_truth':
        print('\tGather Ground-Truth')
        val_dict = general_parse_gt(ibs, test_gid_list=gid_list, **config)

    if output_path is None:
        output_path = abspath(expanduser(join('~', 'Desktop', 'bboxes')))
        ut.ensuredir(output_path)

    filepath_dict = {}
    for gid, image_uuid in zip(gid_list, uuid_list):
        image = ibs.get_images(gid)
        image = _resize(image, t_width=t_width)
        h, w, c = image.shape

        val_list = val_dict[image_uuid]

        for val in val_list:
            xbr = int(np.around(val['xbr'] * w))
            ybr = int(np.around(val['ybr'] * h))
            xtl = int(np.around(val['xtl'] * w))
            ytl = int(np.around(val['ytl'] * h))
            cv2.rectangle(image, (xtl, ytl), (xbr, ybr), (0, 140, 255), 4)

        write_filename = 'bboxes_%d_%s.png' % (gid, version,)
        write_filepath = join(output_path, write_filename)
        print(write_filepath)
        cv2.imwrite(write_filepath, image)

        filepath_dict[gid] = write_filepath

    return filepath_dict


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.other.detectcore
        python -m wbia.other.detectcore --allexamples
        python -m wbia.other.detectcore --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
