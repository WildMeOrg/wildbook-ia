from __future__ import absolute_import, division, print_function, unicode_literals
from ibeis_cnn.ingest_ibeis import get_cnn_classifier_cameratrap_binary_training_images_pytorch
from ibeis.control import controller_inject
from os.path import join, exists
import utool as ut
from ibeis.algo.detect import wic


PYTORCH = True


# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[vulcanfuncs]')


# Must import class before injection
CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))


register_api = controller_inject.get_ibeis_flask_api(__name__)


@register_ibs_method
def vulcan_imageset_train_test_split(ibs, imageset_text_list=None):

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
def vulcan_wic_train(ibs, ensembles=5):
    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', 'NEGATIVE'])

    skip_rate_neg = 1.0 - (1.0 / ensembles)

    weights_path_list = []
    for index in range(ensembles):
        data_path = join(ibs.get_cachedir(), 'extracted-%d' % (index, ))
        output_path = join(ibs.get_cachedir(), 'training', 'classifier-cameratrap-%d' % (index, ))

        extracted_path = get_cnn_classifier_cameratrap_binary_training_images_pytorch(
            ibs,
            pid,
            nid,
            dest_path=data_path,
            skip_rate_neg=skip_rate_neg,
        )
        weights_path = wic.train(extracted_path, output_path)
        weights_path_list.append(weights_path)

    return weights_path_list


@register_ibs_method
def vulcan_wic_deploy(ibs, weights_path_list):
    ensemble_path = join(ibs.get_cachedir(), 'training', 'ensemble')
    ut.ensuredir(ensemble_path)

    archive_path = '%s.tar' % (ensemble_path)
    ensemble_weights_path_list = []

    for index, weights_path in enumerate(sorted(weights_path_list)):
        assert exists(weights_path)
        ensemble_weights_path = join(ensemble_path, 'classifier.%d.weights' % (index, ))
        ut.copy(weights_path, ensemble_weights_path)
        ensemble_weights_path_list.append(ensemble_weights_path)

    ut.archive_files(archive_path, ensemble_weights_path_list, overwrite=True)
    ut.copy(archive_path, '/data/public/models/classifier2.vulcan.tar')
    return archive_path


@register_ibs_method
def vulcan_wic_validate(ibs, model_tag=None, imageset_text_list=None):
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

    test_imgsetid = ibs.add_imagesets('TEST_SET')
    test_gid_list = ibs.get_imageset_gids(test_imgsetid)

    test_tile_list = list(set(tile_list) & set(test_gid_list))

    pid, nid = ibs.get_imageset_imgsetids_from_text(['POSITIVE', 'NEGATIVE'])
    # positive_gid_set = set(ibs.get_imageset_gids(pid))
    # negative_gid_set = set(ibs.get_imageset_gids(nid))

    # test_label_list = []
    # for test_tile in test_tile_list:
    #     if test_tile in positive_gid_set:
    #         test_label_list.append('positive')
    #     elif test_tile in negative_gid_set:
    #         test_label_list.append('negative')
    #     else:
    #         raise ValueError()

    # config = {
    #     'classifier_algo': 'wic',
    #     'classifier_weight_filepath': model_tag,
    # }
    # prediction_list = ibs.depc_image.get_property('classifier', test_tile_list, 'class', config=config)
    # confidence_list = ibs.depc_image.get_property('classifier', test_tile_list, 'score', config=config)
    # confidence_list = [
    #     confidence if prediction == 'positive' else 1.0 - confidence
    #     for prediction, confidence in zip(prediction_list, confidence_list)
    # ]

    # return confidence_list, test_label_list

    ibs.classifier_cameratrap_precision_recall_algo_display(pid, nid, test_gid_list=test_tile_list)


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
    annot_size = int(np.around(np.mean(w_list) + np.std(w_list)))
    patch_size = annot_size // 2

    data_path = join(ibs.get_cachedir(), 'extracted')
    output_path = join(ibs.get_cachedir(), 'training', 'background')

    species = 'elephant_savanna'
    extracted_path = get_background_training_patches2(ibs, species, data_path,
                                                      patch_size=patch_size,
                                                      annot_size=annot_size,
                                                      patch_size_min=0.9,
                                                      patch_size_max=1.1,
                                                      patches_per_annotation=10,
                                                      train_gid_set=train_gid_set,
                                                      visualize=True,
                                                      inside_boundary=False,
                                                      purge=True)

    id_file, X_file, y_file = numpy_processed_directory2(extracted_path)
    model_path = train_background(output_path, X_file, y_file)
    model_state = ut.load_cPkl(model_path)
    assert 'species' not in model_state
    model_state['species'] = species
    save_model(model_state, model_path)

    return model_path


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
