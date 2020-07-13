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
from wbia.control import controller_inject
from os.path import join, exists
import utool as ut

# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[other.detecttrain]')


CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)


@register_ibs_method
def classifier_cameratrap_train(
    ibs, positive_imageset_id, negative_imageset_id, **kwargs
):
    from wbia_cnn.ingest_wbia import get_cnn_classifier_cameratrap_binary_training_images
    from wbia_cnn.process import numpy_processed_directory2
    from wbia_cnn.models.classifier import train_classifier

    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_cnn_classifier_cameratrap_binary_training_images(
        ibs, positive_imageset_id, negative_imageset_id, dest_path=data_path, **kwargs
    )
    id_file, X_file, y_file = numpy_processed_directory2(extracted_path)
    output_path = join(ibs.get_cachedir(), 'training', 'classifier-cameratrap')
    model_path = train_classifier(output_path, X_file, y_file)
    # Return model path
    return model_path


@register_ibs_method
def classifier_cameratrap_densenet_train(
    ibs, positive_imageset_id, negative_imageset_id, ensembles=3, **kwargs
):
    from wbia.other.detectexport import (
        get_cnn_classifier_cameratrap_binary_training_images_pytorch,
    )
    from wbia.algo.detect import densenet

    data_path = join(ibs.get_cachedir(), 'extracted-classifier-cameratrap')
    extracted_path = get_cnn_classifier_cameratrap_binary_training_images_pytorch(
        ibs,
        positive_imageset_id,
        negative_imageset_id,
        dest_path=data_path,
        image_size=densenet.INPUT_SIZE,
        **kwargs,
    )

    weights_path_list = []
    for ensemble_num in range(ensembles):
        args = (ensemble_num,)
        output_path = join(
            ibs.get_cachedir(), 'training', 'classifier-cameratrap-ensemble-%d' % args
        )
        weights_path = densenet.train(extracted_path, output_path, blur=True, flip=True)
        weights_path_list.append(weights_path)

    archive_name = 'classifier.cameratrap.zip'
    archive_path = join(ibs.get_cachedir(), 'training', archive_name)
    ensemble_weights_path_list = []

    for index, weights_path in enumerate(sorted(weights_path_list)):
        assert exists(weights_path)
        ensemble_weights_path = 'classifier.cameratrap.%d.weights' % (index,)
        ut.copy(weights_path, ensemble_weights_path)
        ensemble_weights_path_list.append(ensemble_weights_path)

    ut.archive_files(
        archive_path, ensemble_weights_path_list, overwrite=True, common_prefix=True
    )

    return archive_path


@register_ibs_method
def classifier_multiclass_densenet_train(
    ibs, gid_list, label_list, ensembles=3, **kwargs
):
    """
    >>> import uuid
    >>> manifest_filepath = join(ibs.dbdir, 'flukebook_groundtruth.csv')
    >>> with open(manifest_filepath, 'r') as manifest_file:
    >>>     line_list = manifest_file.readlines()
    >>>
    >>> label_dict = {
    >>>     'Left Dorsal Fin'  : 'left_dorsal_fin',
    >>>     'Right Dorsal Fin' : 'right_dorsal_fin',
    >>>     'Tail Fluke'       : 'tail_fluke',
    >>> }
    >>>
    >>> uuid_list = []
    >>> label_list = []
    >>> for line in line_list:
    >>>     line = line.strip().split(',')
    >>>     assert len(line) == 2
    >>>     uuid_, label_ = line
    >>>     uuid_ = uuid.UUID(uuid_)
    >>>     label_ = label_.strip()
    >>>     print(uuid_, label_)
    >>>     uuid_list.append(uuid_)
    >>>     label_ = label_dict.get(label_, None)
    >>>     assert label_ is not None
    >>>     label_list.append(label_)
    >>>
    >>> gid_list = ibs.get_image_gids_from_uuid(uuid_list)
    >>> assert None not in gid_list
    >>> # archive_path = ibs.classifier_multiclass_densenet_train(gid_list, label_list)
    >>> ibs.classifier2_precision_recall_algo_display(test_gid_list=gid_list, test_label_list=label_list)
    """
    from wbia.other.detectexport import (
        get_cnn_classifier_multiclass_training_images_pytorch,
    )
    from wbia.algo.detect import densenet

    data_path = join(ibs.get_cachedir(), 'extracted-classifier-multiclass')
    extracted_path = get_cnn_classifier_multiclass_training_images_pytorch(
        ibs,
        gid_list,
        label_list,
        dest_path=data_path,
        image_size=densenet.INPUT_SIZE,
        **kwargs,
    )

    weights_path_list = []
    for ensemble_num in range(ensembles):
        args = (ensemble_num,)
        output_path = join(
            ibs.get_cachedir(), 'training', 'classifier-multiclass-ensemble-%d' % args
        )
        weights_path = densenet.train(extracted_path, output_path, blur=True, flip=False)
        weights_path_list.append(weights_path)

    archive_name = 'classifier.multiclass.zip'
    archive_path = join(ibs.get_cachedir(), 'training', archive_name)
    ensemble_weights_path_list = []

    for index, weights_path in enumerate(sorted(weights_path_list)):
        assert exists(weights_path)
        ensemble_weights_path = 'classifier.multiclass.%d.weights' % (index,)
        ut.copy(weights_path, ensemble_weights_path)
        ensemble_weights_path_list.append(ensemble_weights_path)

    ut.archive_files(
        archive_path, ensemble_weights_path_list, overwrite=True, common_prefix=True
    )

    return archive_path


@register_ibs_method
def classifier_binary_train(ibs, species_list, **kwargs):
    from wbia_cnn.ingest_wbia import get_cnn_classifier_binary_training_images
    from wbia_cnn.process import numpy_processed_directory2
    from wbia_cnn.models.classifier import train_classifier
    from wbia_cnn.utils import save_model

    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_cnn_classifier_binary_training_images(
        ibs, species_list, dest_path=data_path, **kwargs
    )
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
def classifier2_train(
    ibs, species_list=None, species_mapping={}, train_gid_set=None, **kwargs
):
    from wbia_cnn.ingest_wbia import get_cnn_classifier2_training_images
    from wbia_cnn.process import numpy_processed_directory3
    from wbia_cnn.models.classifier2 import train_classifier2
    from wbia_cnn.utils import save_model

    if species_list is not None:
        species_list = sorted(species_list)
    data_path = join(ibs.get_cachedir(), 'extracted')
    values = get_cnn_classifier2_training_images(
        ibs,
        species_list,
        category_mapping=species_mapping,
        train_gid_set=train_gid_set,
        dest_path=data_path,
        **kwargs,
    )
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
def canonical_classifier_train(ibs, species, ensembles=3, extracted_path=None, **kwargs):
    from wbia.other.detectexport import (
        get_cnn_classifier_canonical_training_images_pytorch,
    )
    from wbia.algo.detect import densenet

    args = (species,)
    data_path = join(ibs.get_cachedir(), 'extracted-classifier-canonical-%s' % args)
    if extracted_path is None:
        extracted_path = get_cnn_classifier_canonical_training_images_pytorch(
            ibs, species, dest_path=data_path,
        )

    weights_path_list = []
    for ensemble_num in range(ensembles):
        args = (
            species,
            ensemble_num,
        )
        output_path = join(
            ibs.get_cachedir(), 'training', 'classifier-canonical-%s-ensemble-%d' % args
        )
        if exists(output_path):
            ut.delete(output_path)
        weights_path = densenet.train(extracted_path, output_path, blur=False, flip=False)
        weights_path_list.append(weights_path)

    args = (species,)
    output_name = 'classifier.canonical.%s' % args
    ensemble_path = join(ibs.get_cachedir(), 'training', output_name)
    ut.ensuredir(ensemble_path)

    archive_path = '%s.zip' % (ensemble_path)
    ensemble_weights_path_list = []

    for index, weights_path in enumerate(sorted(weights_path_list)):
        assert exists(weights_path)
        ensemble_weights_path = join(
            ensemble_path, 'classifier.canonical.%d.weights' % (index,)
        )
        ut.copy(weights_path, ensemble_weights_path)
        ensemble_weights_path_list.append(ensemble_weights_path)

    ensemble_weights_path_list = [ensemble_path] + ensemble_weights_path_list
    ut.archive_files(
        archive_path, ensemble_weights_path_list, overwrite=True, common_prefix=True
    )

    return archive_path


@register_ibs_method
def canonical_localizer_train(ibs, species, ensembles=3, **kwargs):
    from wbia.other.detectexport import (
        get_cnn_localizer_canonical_training_images_pytorch,
    )
    from wbia.algo.detect import canonical

    args = (species,)
    data_path = join(ibs.get_cachedir(), 'extracted-localizer-canonical-%s' % args)
    extracted_path = get_cnn_localizer_canonical_training_images_pytorch(
        ibs, species, dest_path=data_path,
    )

    weights_path_list = []
    for ensemble_num in range(ensembles):
        args = (
            species,
            ensemble_num,
        )
        output_path = join(
            ibs.get_cachedir(), 'training', 'localizer-canonical-%s-ensemble-%d' % args
        )
        weights_path = canonical.train(extracted_path, output_path)
        weights_path_list.append(weights_path)

    args = (species,)
    output_name = 'localizer.canonical.%s' % args
    ensemble_path = join(ibs.get_cachedir(), 'training', output_name)
    ut.ensuredir(ensemble_path)

    archive_path = '%s.zip' % (ensemble_path)
    ensemble_weights_path_list = []

    for index, weights_path in enumerate(sorted(weights_path_list)):
        assert exists(weights_path)
        ensemble_weights_path = join(
            ensemble_path, 'localizer.canonical.%d.weights' % (index,)
        )
        ut.copy(weights_path, ensemble_weights_path)
        ensemble_weights_path_list.append(ensemble_weights_path)

    ensemble_weights_path_list = [ensemble_path] + ensemble_weights_path_list
    ut.archive_files(
        archive_path, ensemble_weights_path_list, overwrite=True, common_prefix=True
    )

    return archive_path


@register_ibs_method
def localizer_yolo_train(ibs, species_list=None, **kwargs):
    from pydarknet import Darknet_YOLO_Detector

    data_path = ibs.export_to_xml(species_list=species_list, **kwargs)
    output_path = join(ibs.get_cachedir(), 'training', 'localizer')
    ut.ensuredir(output_path)
    dark = Darknet_YOLO_Detector()
    model_path = dark.train(data_path, output_path)
    del dark
    return model_path


def _localizer_lightnet_validate_training_kit(lightnet_training_kit_url):
    # Remove bad files
    delete_path_list = [
        join(lightnet_training_kit_url, '__MACOSX'),
    ]
    for delete_path in delete_path_list:
        if exists(delete_path):
            ut.delete(delete_path)

    # Ensure first-level structure
    bin_path = join(lightnet_training_kit_url, 'bin')
    cfg_path = join(lightnet_training_kit_url, 'cfg')
    data_path = join(lightnet_training_kit_url, 'data')
    weights_path = join(lightnet_training_kit_url, 'darknet19_448.conv.23.pt')
    assert exists(bin_path)
    assert exists(cfg_path)
    assert exists(data_path)
    assert exists(weights_path)

    # Ensure second-level structure
    dataset_py_path = join(bin_path, 'dataset.template.py')
    labels_py_path = join(bin_path, 'labels.template.py')
    test_py_path = join(bin_path, 'test.template.py')
    train_py_path = join(bin_path, 'train.template.py')
    config_py_path = join(cfg_path, 'yolo.template.py')
    assert exists(dataset_py_path)
    assert exists(labels_py_path)
    assert exists(test_py_path)
    assert exists(train_py_path)
    assert exists(config_py_path)


def _localizer_lightnet_template_replace(
    template_filepath, replace_dict, output_filepath=None
):
    if output_filepath is None:
        output_filepath = template_filepath.replace('.template.', '.')
    with open(template_filepath, 'r') as template_file:
        template = ''.join(template_file.readlines())
    for search_str, replace_str in replace_dict.items():
        search_str = str(search_str)
        replace_str = str(replace_str)
        template = template.replace(search_str, replace_str)
    with open(output_filepath, 'w') as output_file:
        output_file.write(template)
    return output_filepath


@register_ibs_method
def localizer_lightnet_train(
    ibs,
    species_list,
    cuda_device='0',
    batches=60000,
    validate_with_accuracy=True,
    deploy_tag=None,
    cleanup=True,
    cleanup_all=True,
    deploy=True,
    cache_species_str=None,
    **kwargs,
):
    from wbia.algo.detect import lightnet
    import subprocess
    import datetime
    import math
    import sys

    assert species_list is not None
    species_list = sorted(species_list)

    lightnet_training_kit_url = lightnet._download_training_kit()
    _localizer_lightnet_validate_training_kit(lightnet_training_kit_url)

    hashstr = ut.random_nonce()[:16]
    if cache_species_str is None:
        cache_species_str = '-'.join(species_list)

    cache_path = join(ibs.cachedir, 'training', 'lightnet')
    ut.ensuredir(cache_path)
    training_instance_folder = 'lightnet-training-%s-%s' % (cache_species_str, hashstr,)
    training_instance_path = join(cache_path, training_instance_folder)
    ut.copy(lightnet_training_kit_url, training_instance_path)

    backup_path = join(training_instance_path, 'backup')
    bin_path = join(training_instance_path, 'bin')
    cfg_path = join(training_instance_path, 'cfg')
    data_path = join(training_instance_path, 'data')
    deploy_path = join(training_instance_path, 'deploy')
    weights_path = join(training_instance_path, 'darknet19_448.conv.23.pt')
    results_path = join(training_instance_path, 'results.txt')
    dataset_py_path = join(bin_path, 'dataset.template.py')
    labels_py_path = join(bin_path, 'labels.template.py')
    test_py_path = join(bin_path, 'test.template.py')
    train_py_path = join(bin_path, 'train.template.py')
    config_py_path = join(cfg_path, 'yolo.template.py')

    ibs.export_to_xml(species_list=species_list, output_path=data_path, **kwargs)

    species_str_list = ['%r' % (species,) for species in species_list]
    species_str = ', '.join(species_str_list)
    replace_dict = {
        '_^_YEAR_^_': str(datetime.datetime.now().year),
        '_^_DATA_ROOT_^_': data_path,
        '_^_SPECIES_MAPPING_^_': species_str,
        '_^_NUM_BATCHES_^_': str(batches),
    }

    dataset_py_path = _localizer_lightnet_template_replace(dataset_py_path, replace_dict)
    labels_py_path = _localizer_lightnet_template_replace(labels_py_path, replace_dict)
    test_py_path = _localizer_lightnet_template_replace(test_py_path, replace_dict)
    train_py_path = _localizer_lightnet_template_replace(train_py_path, replace_dict)
    config_py_path = _localizer_lightnet_template_replace(config_py_path, replace_dict)
    assert exists(dataset_py_path)
    assert exists(labels_py_path)
    assert exists(test_py_path)
    assert exists(train_py_path)
    assert exists(config_py_path)
    assert not exists(backup_path)
    assert not exists(results_path)

    python_exe = sys.executable
    cuda_str = (
        ''
        if cuda_device in [-1, None] or len(cuda_device) == 0
        else 'CUDA_VISIBLE_DEVICES=%s ' % (cuda_device,)
    )

    # Call labels
    call_str = '%s %s' % (python_exe, labels_py_path,)
    print(call_str)
    subprocess.call(call_str, shell=True)

    # Call training
    # Example: CUDA_VISIBLE_DEVICES=X python bin/train.py -c -n cfg/yolo.py -c darknet19_448.conv.23.pt
    args = (
        cuda_str,
        python_exe,
        train_py_path,
        config_py_path,
        backup_path,
        weights_path,
    )
    call_str = '%s%s %s -c -n %s -b %s %s' % args
    print(call_str)
    subprocess.call(call_str, shell=True)
    assert exists(backup_path)

    ut.embed()

    """
    x = (
        'CUDA_VISIBLE_DEVICES=3 ',
        '/home/jason.parham/virtualenv/wildme3.6/bin/python',
        '/data/wbia/WD_Master/_ibsdb/_wbia_cache/training/lightnet/lightnet-training-wilddog-8eb77cb02b66e9d6/bin/test.py',
        '/data/wbia/WD_Master/_ibsdb/_wbia_cache/training/lightnet/lightnet-training-wilddog-8eb77cb02b66e9d6/cfg/yolo.py',
        '/data/wbia/WD_Master/_ibsdb/_wbia_cache/training/lightnet/lightnet-training-wilddog-8eb77cb02b66e9d6/results.txt',
        '/data/wbia/WD_Master/_ibsdb/_wbia_cache/training/lightnet/lightnet-training-wilddog-8eb77cb02b66e9d6/backup',
        True,
        '/data/wbia/WD_Master/_ibsdb/_wbia_cache/training/lightnet/lightnet-training-wilddog-8eb77cb02b66e9d6/deploy',
        True,
        None,
        False,
        True,
        '/data/wbia/WD_Master/_ibsdb/_wbia_cache/training/lightnet/lightnet-training-wilddog-8eb77cb02b66e9d6/bin',
        '/data/wbia/WD_Master/_ibsdb/_wbia_cache/training/lightnet/lightnet-training-wilddog-8eb77cb02b66e9d6/cfg',
        '/data/wbia/WD_Master/_ibsdb/_wbia_cache/training/lightnet/lightnet-training-wilddog-8eb77cb02b66e9d6/data',
        '/data/wbia/WD_Master/_ibsdb/_wbia_cache/training/lightnet/lightnet-training-wilddog-8eb77cb02b66e9d6/darknet19_448.conv.23.pt',
        'wilddog'
    )

    cuda_str, python_exe, test_py_path, config_py_path, results_path, backup_path, validate_with_accuracy, deploy_path, deploy, deploy_tag, cleanup, cleanup_all, bin_path, cfg_path, data_path, weights_path, cache_species_str = x

    call_str = 'CUDA_VISIBLE_DEVICES=3 /home/jason.parham/virtualenv/wildme3.6/bin/python /data/wbia/WD_Master/_ibsdb/_wbia_cache/training/lightnet/lightnet-training-wilddog-8eb77cb02b66e9d6/bin/test.py -c -n /data/wbia/WD_Master/_ibsdb/_wbia_cache/training/lightnet/lightnet-training-wilddog-8eb77cb02b66e9d6/cfg/yolo.py --results /data/wbia/WD_Master/_ibsdb/_wbia_cache/training/lightnet/lightnet-training-wilddog-8eb77cb02b66e9d6/results.txt /data/wbia/WD_Master/_ibsdb/_wbia_cache/training/lightnet/lightnet-training-wilddog-8eb77cb02b66e9d6/backup/*'
     """

    # Call testing
    # Example: CUDA_VISIBLE_DEVICE=X python bin/test.py -c -n cfg/yolo.py
    args = (
        cuda_str,
        python_exe,
        test_py_path,
        config_py_path,
        results_path,
        backup_path,
    )
    call_str = '%s%s %s -c -n %s --results %s %s/*' % args
    print(call_str)
    subprocess.call(call_str, shell=True)
    assert exists(results_path)

    # Validate results
    with open(results_path, 'r') as results_file:
        line_list = results_file.readlines()

    if len(line_list) < 10:
        print('VALIDATION ERROR!')
        ut.embed()

    result_list = []
    for line in line_list:
        # print(line)
        line = line.strip().split(',')
        if len(line) != 3:
            continue
        model_path, loss, accuracy = line
        loss = float(loss)
        accuracy = float(accuracy)
        if math.isnan(accuracy):
            continue
        miss_rate = (100.0 - accuracy) / 100.0
        if validate_with_accuracy:
            assert not math.isnan(miss_rate)
            result = (miss_rate, loss, model_path)
        else:
            assert not math.isnan(loss)
            result = (loss, miss_rate, model_path)
        print('\t%r' % (result,))
        result_list.append(result)
    result_list = sorted(result_list)

    best_result = result_list[0]
    best_model_filepath = best_result[-1]

    # Copy best model, delete the rest
    ut.ensuredir(deploy_path)
    deploy_model_filepath = join(deploy_path, 'detect.lightnet.weights')
    deploy_config_filepath = join(deploy_path, 'detect.lightnet.py')
    ut.copy(best_model_filepath, deploy_model_filepath)
    ut.copy(config_py_path, deploy_config_filepath)

    # Cleanup
    if cleanup:
        ut.delete(backup_path)
        ut.delete(results_path)

        if cleanup_all:
            ut.delete(bin_path)
            ut.delete(cfg_path)
            ut.delete(data_path)
            ut.delete(weights_path)

    # Deploy
    final_path = join('/', 'data', 'public', 'models')
    if deploy:
        assert exists(final_path), 'Cannot deploy the model on this machine'
        if deploy_tag is None:
            deploy_tag = cache_species_str

        counter = 0
        while True:
            final_config_prefix = 'detect.lightnet.%s.v%d' % (deploy_tag, counter,)
            final_config_filename = '%s.py' % (final_config_prefix,)
            final_config_filepath = join(final_path, final_config_filename)
            if not exists(final_config_filepath):
                break
            counter += 1

        final_model_filename = '%s.weights' % (final_config_prefix,)
        final_model_filepath = join(final_path, final_model_filename)

        assert not exists(final_model_filepath)
        assert not exists(final_config_filepath)

        ut.copy(deploy_model_filepath, final_model_filepath)
        ut.copy(deploy_config_filepath, final_config_filepath)

        retval = (
            final_model_filepath,
            final_config_filepath,
        )
    else:
        retval = (
            deploy_model_filepath,
            deploy_config_filepath,
        )

    return retval


def validate_model(
    cuda_str,
    python_exe,
    test_py_path,
    config_py_path,
    results_path,
    backup_path,
    validate_with_accuracy,
    deploy_path,
    deploy,
    deploy_tag,
    cleanup,
    cleanup_all,
    bin_path,
    cfg_path,
    data_path,
    weights_path,
    cache_species_str,
):
    import subprocess

    # Call testing
    # Example: CUDA_VISIBLE_DEVICE=X python bin/test.py -c -n cfg/yolo.py
    args = (
        cuda_str,
        python_exe,
        test_py_path,
        config_py_path,
        results_path,
        backup_path,
    )
    call_str = '%s%s %s -c -n %s --results %s %s/*' % args
    print(call_str)
    subprocess.call(call_str, shell=True)
    assert exists(results_path)

    # Validate results
    with open(results_path, 'r') as results_file:
        line_list = results_file.readlines()

    if len(line_list) < 10:
        print('VALIDATION ERROR!')
        ut.embed()

    result_list = []
    for line in line_list:
        print(line)
        line = line.strip().split(',')
        if len(line) != 3:
            continue
        model_path, loss, accuracy = line
        loss = float(loss)
        accuracy = float(accuracy)
        miss_rate = (100.0 - accuracy) / 100.0
        if validate_with_accuracy:
            result = (miss_rate, loss, model_path)
        else:
            result = (loss, miss_rate, model_path)
        print('\t%r' % (result,))
        result_list.append(result)
    result_list = sorted(result_list)

    best_result = result_list[0]
    best_model_filepath = best_result[-1]

    # Copy best model, delete the rest
    ut.ensuredir(deploy_path)
    deploy_model_filepath = join(deploy_path, 'detect.lightnet.weights')
    deploy_config_filepath = join(deploy_path, 'detect.lightnet.py')
    ut.copy(best_model_filepath, deploy_model_filepath)
    ut.copy(config_py_path, deploy_config_filepath)

    # Cleanup
    if cleanup:
        ut.delete(backup_path)
        ut.delete(results_path)

        if cleanup_all:
            ut.delete(bin_path)
            ut.delete(cfg_path)
            ut.delete(data_path)
            ut.delete(weights_path)

    # Deploy
    final_path = join('/', 'data', 'public', 'models')
    if deploy:
        assert exists(final_path), 'Cannot deploy the model on this machine'
        if deploy_tag is None:
            deploy_tag = cache_species_str

        counter = 0
        while True:
            final_config_prefix = 'detect.lightnet.%s.v%d' % (deploy_tag, counter,)
            final_config_filename = '%s.py' % (final_config_prefix,)
            final_config_filepath = join(final_path, final_config_filename)
            if not exists(final_config_filepath):
                break
            counter += 1

        final_model_filename = '%s.weights' % (final_config_prefix,)
        final_model_filepath = join(final_path, final_model_filename)

        assert not exists(final_model_filepath)
        assert not exists(final_config_filepath)

        ut.copy(deploy_model_filepath, final_model_filepath)
        ut.copy(deploy_config_filepath, final_config_filepath)

        retval = (
            final_model_filepath,
            final_config_filepath,
        )
    else:
        retval = (
            deploy_model_filepath,
            deploy_config_filepath,
        )

    return retval


@register_ibs_method
def labeler_train_wbia_cnn(
    ibs, species_list=None, species_mapping=None, viewpoint_mapping=None, **kwargs
):
    from wbia_cnn.ingest_wbia import get_cnn_labeler_training_images
    from wbia_cnn.process import numpy_processed_directory2
    from wbia_cnn.models.labeler import train_labeler
    from wbia_cnn.utils import save_model

    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_cnn_labeler_training_images(
        ibs,
        data_path,
        category_list=species_list,
        category_mapping=species_mapping,
        viewpoint_mapping=viewpoint_mapping,
        **kwargs,
    )
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


@register_ibs_method
def labeler_train(
    ibs,
    species_list=None,
    species_mapping=None,
    viewpoint_mapping=None,
    ensembles=3,
    **kwargs,
):
    from wbia.other.detectexport import get_cnn_labeler_training_images_pytorch
    from wbia.algo.detect import densenet

    species = '-'.join(species_list)
    args = (species,)
    data_path = join(ibs.get_cachedir(), 'extracted-labeler-%s' % args)
    extracted_path = get_cnn_labeler_training_images_pytorch(
        ibs,
        category_list=species_list,
        category_mapping=species_mapping,
        viewpoint_mapping=viewpoint_mapping,
        dest_path=data_path,
        **kwargs,
    )

    weights_path_list = []
    for ensemble_num in range(ensembles):
        args = (
            species,
            ensemble_num,
        )
        output_path = join(
            ibs.get_cachedir(), 'training', 'labeler-%s-ensemble-%d' % args
        )
        if exists(output_path):
            ut.delete(output_path)
        weights_path = densenet.train(extracted_path, output_path, blur=False, flip=False)
        weights_path_list.append(weights_path)

    args = (species,)
    output_name = 'labeler.%s' % args
    ensemble_path = join(ibs.get_cachedir(), 'training', output_name)
    ut.ensuredir(ensemble_path)

    archive_path = '%s.zip' % (ensemble_path)
    ensemble_weights_path_list = []

    for index, weights_path in enumerate(sorted(weights_path_list)):
        assert exists(weights_path)
        ensemble_weights_path = join(ensemble_path, 'labeler.%d.weights' % (index,))
        ut.copy(weights_path, ensemble_weights_path)
        ensemble_weights_path_list.append(ensemble_weights_path)

    ensemble_weights_path_list = [ensemble_path] + ensemble_weights_path_list
    ut.archive_files(
        archive_path, ensemble_weights_path_list, overwrite=True, common_prefix=True
    )

    return archive_path


# @register_ibs_method
# def qualifier_train(ibs, **kwargs):
#     from wbia_cnn.ingest_wbia import get_cnn_qualifier_training_images
#     from wbia_cnn.process import numpy_processed_directory2
#     from wbia_cnn.models.qualifier import train_qualifier
#     data_path = join(ibs.get_cachedir(), 'extracted')
#     extracted_path = get_cnn_qualifier_training_images(ibs, data_path, **kwargs)
#     id_file, X_file, y_file = numpy_processed_directory2(extracted_path)
#     output_path = join(ibs.get_cachedir(), 'training', 'qualifier')
#     model_path = train_qualifier(output_path, X_file, y_file)
#     return model_path


@register_ibs_method
def detector_train(ibs):
    results = ibs.localizer_yolo_train()
    localizer_weight_path, localizer_config_path, localizer_class_path = results
    classifier_model_path = ibs.classifier_binary_train()
    labeler_model_path = ibs.labeler_train()
    output_path = join(ibs.get_cachedir(), 'training', 'detector')
    ut.ensuredir(output_path)
    ut.copy(localizer_weight_path, join(output_path, 'localizer.weights'))
    ut.copy(localizer_config_path, join(output_path, 'localizer.config'))
    ut.copy(localizer_class_path, join(output_path, 'localizer.classes'))
    ut.copy(classifier_model_path, join(output_path, 'classifier.npy'))
    ut.copy(labeler_model_path, join(output_path, 'labeler.npy'))


@register_ibs_method
def background_train(ibs, species, train_gid_set=None, global_limit=500000, **kwargs):
    from wbia_cnn.ingest_wbia import get_background_training_patches2
    from wbia_cnn.process import numpy_processed_directory2
    from wbia_cnn.models.background import train_background
    from wbia_cnn.utils import save_model

    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_background_training_patches2(
        ibs,
        species,
        data_path,
        patch_size=50,
        train_gid_set=train_gid_set,
        global_limit=global_limit,
        **kwargs,
    )
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
    from wbia_cnn.ingest_wbia import get_aoi_training_data
    from wbia_cnn.process import numpy_processed_directory4
    from wbia_cnn.models.aoi import train_aoi
    from wbia_cnn.utils import save_model

    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_aoi_training_data(
        ibs, data_path, target_species_list=species_list
    )
    id_file, X_file, y_file = numpy_processed_directory4(extracted_path)
    output_path = join(ibs.get_cachedir(), 'training', 'aoi')
    model_path = train_aoi(output_path, X_file, y_file)
    model_state = ut.load_cPkl(model_path)
    assert 'species_list' not in model_state
    model_state['species_list'] = species_list
    save_model(model_state, model_path)
    return model_path


@register_ibs_method
def aoi2_train(ibs, species_list=None, train_gid_list=None, purge=True, cache=False):
    from wbia_cnn.ingest_wbia import get_aoi2_training_data
    from wbia_cnn.process import numpy_processed_directory5
    from wbia_cnn.models.aoi2 import train_aoi2
    from wbia_cnn.utils import save_model

    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_aoi2_training_data(
        ibs,
        dest_path=data_path,
        target_species_list=species_list,
        train_gid_list=train_gid_list,
        purge=purge,
        cache=cache,
    )
    id_file, X_file, y_file = numpy_processed_directory5(extracted_path)
    output_path = join(ibs.get_cachedir(), 'training', 'aoi2')
    model_path = train_aoi2(output_path, X_file, y_file)
    model_state = ut.load_cPkl(model_path)
    assert 'species_list' not in model_state
    model_state['species_list'] = species_list
    save_model(model_state, model_path)
    return model_path


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.other.detecttrain
        python -m wbia.other.detecttrain --allexamples
        python -m wbia.other.detecttrain --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
