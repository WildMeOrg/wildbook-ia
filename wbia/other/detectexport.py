# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut


# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[other.detectexport]')


def get_cnn_classifier_cameratrap_binary_training_images_pytorch(
    ibs,
    positive_imageset_id,
    negative_imageset_id,
    dest_path=None,
    valid_rate=0.2,
    image_size=224,
    purge=True,
    skip_rate=0.0,
    skip_rate_pos=0.0,
    skip_rate_neg=0.0,
):
    from os.path import join, expanduser
    import random
    import cv2

    if dest_path is None:
        dest_path = expanduser(join('~', 'Desktop', 'extracted'))

    name = 'classifier-cameratrap-pytorch'
    dbname = ibs.dbname
    name_path = join(dest_path, name)
    train_path = join(name_path, 'train')
    valid_path = join(name_path, 'val')

    train_pos_path = join(train_path, 'positive')
    train_neg_path = join(train_path, 'negative')
    valid_pos_path = join(valid_path, 'positive')
    valid_neg_path = join(valid_path, 'negative')

    if purge:
        ut.delete(name_path)

    ut.ensuredir(name_path)
    ut.ensuredir(train_path)
    ut.ensuredir(valid_path)

    ut.ensuredir(train_pos_path)
    ut.ensuredir(train_neg_path)
    ut.ensuredir(valid_pos_path)
    ut.ensuredir(valid_neg_path)

    train_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TRAIN_SET'))
    )

    positive_gid_set = set(ibs.get_imageset_gids(positive_imageset_id))
    negative_gid_set = set(ibs.get_imageset_gids(negative_imageset_id))

    candidate_gid_set = positive_gid_set | negative_gid_set
    candidate_gid_set = train_gid_set & candidate_gid_set

    for gid in candidate_gid_set:
        # args = (gid, )
        # print('Processing GID: %r' % args)

        if skip_rate > 0.0 and random.uniform(0.0, 1.0) <= skip_rate:
            print('\t Skipping - Sampling')
            continue

        if gid in positive_gid_set:
            category = 'positive'
        elif gid in negative_gid_set:
            category = 'negative'
        else:
            print('\t Skipping - No Label')
            continue

        if (
            skip_rate_pos > 0.0
            and category == 'positive'
            and random.uniform(0.0, 1.0) <= skip_rate_pos
        ):
            print('\t Skipping Positive')
            continue

        if (
            skip_rate_neg > 0.0
            and category == 'negative'
            and random.uniform(0.0, 1.0) <= skip_rate_neg
        ):
            print('\t Skipping Negative')
            continue

        is_valid = random.uniform(0.0, 1.0) < valid_rate

        if category == 'positive':
            dest_path = valid_pos_path if is_valid else train_pos_path
        elif category == 'negative':
            dest_path = valid_neg_path if is_valid else train_neg_path
        else:
            raise ValueError()

        image = ibs.get_images(gid)
        image_ = cv2.resize(
            image, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4
        )

        values = (
            dbname,
            gid,
        )
        patch_filename = '%s_image_gid_%s.png' % values
        patch_filepath = join(dest_path, patch_filename)
        cv2.imwrite(patch_filepath, image_)

    return name_path


def get_cnn_classifier_multiclass_training_images_pytorch(
    ibs,
    gid_list,
    label_list,
    dest_path=None,
    valid_rate=0.2,
    image_size=224,
    purge=True,
    skip_rate=0.0,
):
    from os.path import join, expanduser
    import random
    import cv2

    if dest_path is None:
        dest_path = expanduser(join('~', 'Desktop', 'extracted'))

    name = 'classifier-multiclass-pytorch'
    dbname = ibs.dbname
    name_path = join(dest_path, name)
    train_path = join(name_path, 'train')
    valid_path = join(name_path, 'val')

    label_set = sorted(set(label_list))

    train_dict = {}
    valid_dict = {}
    for label in label_set:
        assert label not in train_dict
        assert label not in valid_dict
        train_dict[label] = join(train_path, label)
        valid_dict[label] = join(valid_path, label)

    if purge:
        ut.delete(name_path)

    ut.ensuredir(name_path)
    ut.ensuredir(train_path)
    ut.ensuredir(valid_path)

    for label in label_set:
        ut.ensuredir(train_dict[label])
        ut.ensuredir(valid_dict[label])

    train_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TRAIN_SET'))
    )

    for gid, label in zip(gid_list, label_list):
        if gid not in train_gid_set:
            continue

        args = (gid,)
        print('Processing GID: %r' % args)

        if skip_rate > 0.0 and random.uniform(0.0, 1.0) <= skip_rate:
            print('\t Skipping - Sampling')
            continue

        is_valid = random.uniform(0.0, 1.0) < valid_rate
        dest_dict = valid_dict if is_valid else train_dict
        dest_path = dest_dict[label]

        image = ibs.get_images(gid)
        image_ = cv2.resize(
            image, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4
        )

        values = (
            dbname,
            gid,
        )
        patch_filename = '%s_image_gid_%s.png' % values
        patch_filepath = join(dest_path, patch_filename)
        cv2.imwrite(patch_filepath, image_)

    return name_path


def get_cnn_classifier_canonical_training_images_pytorch(
    ibs,
    species,
    dest_path=None,
    valid_rate=0.2,
    image_size=224,
    purge=True,
    skip_rate=0.0,
    skip_rate_pos=0.0,
    skip_rate_neg=0.0,
):
    from os.path import join, expanduser, exists
    import random
    import cv2

    if dest_path is None:
        dest_path = expanduser(join('~', 'Desktop', 'extracted'))

    name = 'classifier-canonical-pytorch'
    dbname = ibs.dbname
    name_path = join(dest_path, name)
    train_path = join(name_path, 'train')
    valid_path = join(name_path, 'val')

    train_pos_path = join(train_path, 'positive')
    train_neg_path = join(train_path, 'negative')
    valid_pos_path = join(valid_path, 'positive')
    valid_neg_path = join(valid_path, 'negative')

    if purge:
        ut.delete(name_path)

    ut.ensuredir(name_path)
    ut.ensuredir(train_path)
    ut.ensuredir(valid_path)

    ut.ensuredir(train_pos_path)
    ut.ensuredir(train_neg_path)
    ut.ensuredir(valid_pos_path)
    ut.ensuredir(valid_neg_path)

    train_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TRAIN_SET'))
    )
    aid_list = ut.flatten(ibs.get_image_aids(train_gid_set))
    aid_list = ibs.filter_annotation_set(aid_list, species=species)
    flag_list = ibs.get_annot_canonical(aid_list)

    bool_list = [flag is not None for flag in flag_list]
    aid_list = ut.compress(aid_list, bool_list)
    flag_list = ut.compress(flag_list, bool_list)

    config = {
        'dim_size': (image_size, image_size),
        'resize_dim': 'wh',
    }
    chip_list = ibs.depc_annot.get_property('chips', aid_list, 'img', config=config)
    for aid, chip, flag in zip(aid_list, chip_list, flag_list):
        args = (aid,)
        print('Processing AID: %r' % args)

        if skip_rate > 0.0 and random.uniform(0.0, 1.0) <= skip_rate:
            print('\t Skipping - Sampling')
            continue

        assert flag is not None

        if flag:
            category = 'positive'
        else:
            category = 'negative'

        if (
            skip_rate_pos > 0.0
            and category == 'positive'
            and random.uniform(0.0, 1.0) <= skip_rate_pos
        ):
            print('\t Skipping Positive')
            continue

        if (
            skip_rate_neg > 0.0
            and category == 'negative'
            and random.uniform(0.0, 1.0) <= skip_rate_neg
        ):
            print('\t Skipping Negative')
            continue

        is_valid = random.uniform(0.0, 1.0) < valid_rate

        if category == 'positive':
            dest_path = valid_pos_path if is_valid else train_pos_path
        elif category == 'negative':
            dest_path = valid_neg_path if is_valid else train_neg_path
        else:
            raise ValueError()

        index = 0
        while True:
            values = (
                dbname,
                aid,
                index,
            )
            patch_filename = '%s_image_aid_%s_%d.png' % values
            patch_filepath = join(dest_path, patch_filename)
            if not exists(patch_filepath):
                break
            index += 1

        cv2.imwrite(patch_filepath, chip)

    return name_path


def get_cnn_localizer_canonical_training_images_pytorch(
    ibs,
    species,
    dest_path=None,
    valid_rate=0.2,
    image_size=224,
    purge=True,
    skip_rate=0.0,
):
    from os.path import join, expanduser, exists
    from wbia.other.detectfuncs import _canonical_get_boxes
    import random
    import cv2

    if dest_path is None:
        dest_path = expanduser(join('~', 'Desktop', 'extracted'))

    name = 'localizer-canonical-pytorch'
    dbname = ibs.dbname
    name_path = join(dest_path, name)
    train_path = join(name_path, 'train')
    valid_path = join(name_path, 'val')

    if purge:
        ut.delete(name_path)

    ut.ensuredir(name_path)
    ut.ensuredir(train_path)
    ut.ensuredir(valid_path)

    train_gid_set = set(
        ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TRAIN_SET'))
    )
    train_gid_list = list(train_gid_set)
    aid_list_, bbox_list = _canonical_get_boxes(ibs, train_gid_list, species)

    config = {
        'dim_size': (image_size, image_size),
        'resize_dim': 'wh',
    }
    chip_list = ibs.depc_annot.get_property('chips', aid_list_, 'img', config=config)
    for aid, chip, bbox in zip(aid_list_, chip_list, bbox_list):
        args = (aid,)
        print('Processing AID: %r' % args)

        if skip_rate > 0.0 and random.uniform(0.0, 1.0) <= skip_rate:
            print('\t Skipping - Sampling')
            continue

        is_valid = random.uniform(0.0, 1.0) < valid_rate
        dest_path = valid_path if is_valid else train_path

        index = 0
        while True:
            values = (
                dbname,
                aid,
                index,
            )
            patch_filename = '%s_image_aid_%s_%d.png' % values
            patch_filepath = join(dest_path, patch_filename)
            if not exists(patch_filepath):
                break
            index += 1

        index = 0
        while True:
            values = (
                dbname,
                aid,
                index,
            )
            label_filename = '%s_image_aid_%s_%d.csv' % values
            label_filepath = join(dest_path, label_filename)
            if not exists(label_filepath):
                break
            index += 1

        cv2.imwrite(patch_filepath, chip)
        with open(label_filepath, 'w') as label_file:
            bbox = list(bbox)
            for index in range(len(bbox)):
                bbox[index] = '%0.08f' % (bbox[index],)
            label_file.write('%s\n' % (','.join(bbox),))

    return name_path


def get_cnn_labeler_training_images_pytorch(
    ibs,
    dest_path=None,
    image_size=224,
    category_list=None,
    min_examples=10,
    category_mapping=None,
    viewpoint_mapping=None,
    purge=True,
    strict=True,
    skip_rate=0.0,
    valid_rate=0.2,
    use_axis_aligned_chips=False,
    train_gid_set=None,
):
    from os.path import join, expanduser, exists
    import random
    import cv2

    if dest_path is None:
        dest_path = expanduser(join('~', 'Desktop', 'extracted'))

    name = 'labeler-pytorch'
    dbname = ibs.dbname
    name_path = join(dest_path, name)
    train_path = join(name_path, 'train')
    valid_path = join(name_path, 'val')

    if purge:
        ut.delete(name_path)

    ut.ensuredir(name_path)
    ut.ensuredir(train_path)
    ut.ensuredir(valid_path)

    print('category mapping = %s' % (ut.repr3(category_mapping),))
    print('viewpoint mapping = %s' % (ut.repr3(viewpoint_mapping),))

    # train_gid_set = ibs.get_valid_gids()
    if train_gid_set is None:
        train_gid_set = set(
            ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TRAIN_SET'))
        )

    aids_list = ibs.get_image_aids(train_gid_set)
    # bboxes_list = [ ibs.get_annot_bboxes(aid_list) for aid_list in aids_list ]
    # aid_list = ibs.get_valid_aids()
    aid_list = ut.flatten(aids_list)
    # import random
    # random.shuffle(aid_list)
    # aid_list = sorted(aid_list[:100])
    species_list = ibs.get_annot_species_texts(aid_list)
    if category_mapping is not None:
        species_list = [
            category_mapping.get(species, species) for species in species_list
        ]
    species_set = set(species_list)
    yaw_list = ibs.get_annot_viewpoints(aid_list)

    if category_list is None:
        category_list = sorted(list(species_set))
        undesired_list = [
            'unspecified_animal',
            ibs.get_species_nice(ibs.const.UNKNOWN_SPECIES_ROWID),
        ]
        for undesired_species in undesired_list:
            if undesired_species in category_list:
                category_list.remove(undesired_species)
    category_set = set(category_list)

    # Filter the tup_list based on the requested categories
    tup_list = list(zip(aid_list, species_list, yaw_list))
    old_len = len(tup_list)
    tup_list = [
        (aid, species, viewpoint_mapping.get(species, {}).get(yaw, yaw),)
        for aid, species, yaw in tup_list
        if species in category_set
    ]
    new_len = len(tup_list)
    print('Filtered annotations: keep %d / original %d' % (new_len, old_len,))

    # Skip any annotations that are of the wanted category and don't have a specified viewpoint
    counter = 0
    seen_dict = {}
    yaw_dict = {}
    for tup in tup_list:
        aid, species, yaw = tup
        # Keep track of the number of overall instances
        if species not in seen_dict:
            seen_dict[species] = 0
        seen_dict[species] += 1
        # Keep track of yaws that aren't None
        if yaw is not None:
            if species not in yaw_dict:
                yaw_dict[species] = {}
            if yaw not in yaw_dict[species]:
                yaw_dict[species][yaw] = 0
            yaw_dict[species][yaw] += 1
        else:
            counter += 1

    # Get the list of species that do not have enough viewpoint examples for training
    invalid_seen_set = set([])
    invalid_yaw_set = set([])
    for species in seen_dict:
        # Check that the number of instances is above the min_examples
        if seen_dict[species] < min_examples:
            invalid_seen_set.add(species)
            continue
        # If the species has viewpoints, check them as well
        if strict:
            if species in yaw_dict:
                # Check that all viewpoints exist
                # if len(yaw_dict[species]) < 8:
                #     invalid_yaw_set.add(species)
                #     continue
                # Check that all viewpoints have a minimum number of instances
                for yaw in yaw_dict[species]:
                    # assert yaw in ibs.const.VIEWTEXT_TO_YAW_RADIANS
                    if yaw_dict[species][yaw] < min_examples:
                        invalid_yaw_set.add(species)
                        continue
            else:
                invalid_yaw_set.add(species)
                continue

    print('Null yaws: %d' % (counter,))
    valid_seen_set = category_set - invalid_seen_set
    valid_yaw_set = valid_seen_set - invalid_yaw_set
    print('Requested categories:')
    category_set = sorted(category_set)
    ut.print_list(category_set)
    # print('Invalid yaw categories:')
    # ut.print_list(sorted(invalid_yaw_set))
    # print('Valid seen categories:')
    # ut.print_list(sorted(valid_seen_set))
    print('Valid yaw categories:')
    valid_yaw_set = sorted(valid_yaw_set)
    ut.print_list(valid_yaw_set)
    print('Invalid seen categories (could not fulfill request):')
    invalid_seen_set = sorted(invalid_seen_set)
    ut.print_list(invalid_seen_set)

    skipped_yaw = 0
    skipped_seen = 0
    aid_list_ = []
    category_list_ = []
    for tup in tup_list:
        aid, species, yaw = tup
        if species in valid_yaw_set:
            # If the species is valid, but this specific annotation has no yaw, skip it
            if yaw is None:
                skipped_yaw += 1
                continue
            category = '%s:%s' % (species, yaw,)
        elif species in valid_seen_set:
            category = '%s' % (species,)
        else:
            skipped_seen += 1
            continue
        aid_list_.append(aid)
        category_list_.append(category)
    print('Skipped Yaw:  skipped %d / total %d' % (skipped_yaw, len(tup_list),))
    print('Skipped Seen: skipped %d / total %d' % (skipped_seen, len(tup_list),))

    for category in sorted(set(category_list_)):
        print('Making folder for %r' % (category,))
        ut.ensuredir(join(train_path, category))
        ut.ensuredir(join(valid_path, category))

    config = {
        'dim_size': (image_size, image_size),
        'resize_dim': 'wh',
        'axis_aligned': use_axis_aligned_chips,
    }
    chip_list_ = ibs.depc_annot.get_property('chips', aid_list_, 'img', config=config)

    # Get training data
    label_list = []
    for aid, chip, category in zip(aid_list_, chip_list_, category_list_):

        args = (aid,)
        print('Processing AID: %r' % args)

        if skip_rate > 0.0 and random.uniform(0.0, 1.0) <= skip_rate:
            print('\t Skipping')
            continue

        is_valid = random.uniform(0.0, 1.0) < valid_rate
        dest_path = valid_path if is_valid else train_path
        raw_path = join(dest_path, category)
        assert exists(dest_path)

        # Compute data
        values = (
            dbname,
            aid,
        )
        patch_filename = '%s_annot_aid_%s.png' % values
        patch_filepath = join(raw_path, patch_filename)
        cv2.imwrite(patch_filepath, chip)

        # Compute label
        label = '%s,%s' % (patch_filename, category,)
        label_list.append(label)

    print('Using labels for labeler training:')
    print(ut.repr3(ut.dict_hist(category_list_)))

    return name_path


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.other.detectexport
        python -m wbia.other.detectexport --allexamples
        python -m wbia.other.detectexport --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
