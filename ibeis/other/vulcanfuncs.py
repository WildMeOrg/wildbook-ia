from __future__ import absolute_import, division, print_function, unicode_literals
from ibeis_cnn.ingest_ibeis import get_cnn_classifier_cameratrap_binary_training_images_pytorch
from ibeis.control import controller_inject
from ibeis.algo.detect import wic
from os.path import join, exists
import numpy as np
import utool as ut
import cv2


PYTORCH = True


# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[vulcanfuncs]')


# Must import class before injection
CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))


register_api = controller_inject.get_ibeis_flask_api(__name__)


@register_ibs_method
def vulcan_get_valid_tile_rowids(ibs, imageset_text_list=None):
    if imageset_text_list is None:
        imageset_text_list = [
            'elephant',
            'RR18_BIG_2015_09_23_R_AM',
            'TA24_TPM_L_2016-10-30-A',
            'TA24_TPM_R_2016-10-30-A',
        ]

    imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(imageset_text_list)
    gids_list = ibs.get_imageset_gids(imageset_rowid_list)
    gid_list = ut.flatten(gids_list)

    config = {
        'tile_width':   256,
        'tile_height':  256,
        'tile_overlap': 64,
    }
    tiles_list = ibs.compute_tiles(gid_list=gid_list, **config)
    tile_list = ut.flatten(tiles_list)

    return tile_list


@register_ibs_method
def vulcan_imageset_train_test_split(ibs, **kwargs):
    tile_list = ibs.vulcan_get_valid_tile_rowids(**kwargs)

    aids_list = ibs.get_vulcan_image_tile_aids(tile_list)
    species_list_list = list(map(ibs.get_annot_species_texts, aids_list))
    flag_list = [
        'elephant_savanna' in species_list
        for species_list in species_list_list
    ]

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

    x = list(map(len, ibs.get_imageset_gids([pid, nid])))
    num_pos, num_neg = x

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

    ibs.set_image_imgsetids(tile_train_list, [train_imgsetid] * len(tile_train_list))
    ibs.set_image_imgsetids(tile_test_list, [test_imgsetid] * len(tile_test_list))

    return tile_list


@register_ibs_method
def vulcan_wic_train(ibs, ensembles=5, negative_imageset_text='NEGATIVE', round_num=0, hashstr=None):
    if hashstr is None:
        hashstr = ut.random_nonce()[:8]
    print('Using hashstr=%r' % (hashstr, ))

    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', negative_imageset_text])

    skip_rate_neg = 1.0 - (1.0 / ensembles)

    weights_path_list = []
    for index in range(ensembles):
        args = (hashstr, round_num, index, )
        data_path = join(ibs.get_cachedir(), 'extracted-%s-%d-%d' % args)
        output_path = join(ibs.get_cachedir(), 'training', 'classifier-cameratrap-%s-%d-%d' % args)

        extracted_path = get_cnn_classifier_cameratrap_binary_training_images_pytorch(
            ibs,
            pid,
            nid,
            dest_path=data_path,
            skip_rate_neg=skip_rate_neg,
        )
        weights_path = wic.train(extracted_path, output_path)
        weights_path_list.append(weights_path)

    return weights_path_list, hashstr


@register_ibs_method
def vulcan_wic_deploy(ibs, weights_path_list, hashstr, round_num=0):
    args = (hashstr, round_num, )
    ensemble_name = 'ensemble-hashstr-%s-round-%r' % args
    ensemble_path = join(ibs.get_cachedir(), 'training', ensemble_name)
    ut.ensuredir(ensemble_path)

    archive_path = '%s.tar' % (ensemble_path)
    ensemble_weights_path_list = []

    for index, weights_path in enumerate(sorted(weights_path_list)):
        assert exists(weights_path)
        ensemble_weights_path = join(ensemble_path, 'classifier.%d.weights' % (index, ))
        ut.copy(weights_path, ensemble_weights_path)
        ensemble_weights_path_list.append(ensemble_weights_path)

    ut.archive_files(archive_path, ensemble_weights_path_list, overwrite=True, common_prefix=True)

    output_path = '/data/public/models/classifier2.vulcan.%s.%d.tar' % args
    ut.copy(archive_path, output_path)

    return archive_path


@register_ibs_method
def vulcan_wic_test(ibs, test_tile_list, model_tag=None):
    config = {
        'classifier_algo': 'wic',
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
def vulcan_wic_boost(ibs, model_tag=None, ensembles=5, round_num=1,
                     confidence_thresh=0.1, **kwargs):

    ut.embed()

    assert round_num >= 1
    all_tile_set = set(ibs.vulcan_get_valid_tile_rowids(**kwargs))

    train_gid_set = set(ibs.get_imageset_gids(ibs.get_imageset_imgsetids_from_text('TRAIN_SET')))
    train_gid_set = all_tile_set & train_gid_set

    nid, = ibs.get_imageset_imgsetids_from_text(['NEGATIVE'])
    negative_gid_set = set(ibs.get_imageset_gids(nid))
    negative_gid_set = negative_gid_set & train_gid_set

    test_tile_list = list(negative_gid_set)
    confidence_list = ibs.vulcan_wic_test(test_tile_list, model_tag=model_tag)

    boost_imageset_text = 'NEGATIVE-BOOST-%d' % (round_num, )
    boost_id, = ibs.get_imageset_imgsetids_from_text([boost_imageset_text])
    gid_all_list = ibs.get_valid_gids(is_tile=None)
    ibs.unrelate_images_and_imagesets(gid_all_list, [boost_id] * len(gid_all_list))

    flag_list = [confidence >= confidence_thresh for confidence in confidence_list]
    test_tile_list = ut.compress(test_tile_list, flag_list)
    print('Using %d / %d boosted negatives' % (len(test_tile_list), len(flag_list), ))
    ibs.set_image_imagesettext(test_tile_list, [boost_imageset_text] * len(test_tile_list))

    weights_path_list = ibs.vulcan_wic_train(ensembles=ensembles,
                                             negative_imageset_text=boost_imageset_text,
                                             round_num=round_num)

    return weights_path_list


@register_ibs_method
def vulcan_wic_validate(ibs, model_tag=None, **kwargs):
    tile_list = ibs.vulcan_get_valid_tile_rowids(**kwargs)

    test_imgsetid = ibs.add_imagesets('TEST_SET')
    test_gid_list = ibs.get_imageset_gids(test_imgsetid)

    test_tile_list = list(set(tile_list) & set(test_gid_list))

    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', 'NEGATIVE'])

    # return confidence_list, test_label_list
    config_list = [
        {'label': 'ELPH WIC Ensemble', 'classifier_algo': 'wic', 'classifier_weight_filepath': 'vulcan'},
        {'label': 'ELPH WIC 0',        'classifier_algo': 'wic', 'classifier_weight_filepath': 'vulcan:0'},
        {'label': 'ELPH WIC 1',        'classifier_algo': 'wic', 'classifier_weight_filepath': 'vulcan:1'},
        {'label': 'ELPH WIC 2',        'classifier_algo': 'wic', 'classifier_weight_filepath': 'vulcan:2'},
        {'label': 'ELPH WIC 3',        'classifier_algo': 'wic', 'classifier_weight_filepath': 'vulcan:3'},
        {'label': 'ELPH WIC 4',        'classifier_algo': 'wic', 'classifier_weight_filepath': 'vulcan:4'},
    ]

    ibs.classifier_cameratrap_precision_recall_algo_display(pid, nid, test_gid_list=test_tile_list,
                                                            config_list=config_list, offset_black=1)


@register_ibs_method
def vulcan_background_train(ibs):
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

    species = 'elephant_savanna'
    extracted_path = get_background_training_patches2(ibs, species, data_path,
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
    model_state['species'] = species
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

    tile_list = ibs.vulcan_get_valid_tile_rowids(**kwargs)

    test_imgsetid = ibs.add_imagesets('TEST_SET')
    test_gid_list = ibs.get_imageset_gids(test_imgsetid)

    test_tile_list = list(set(tile_list) & set(test_gid_list))

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
