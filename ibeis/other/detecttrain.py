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
from os.path import join
import utool as ut
from ibeis.control import controller_inject

# Inject utool functions
(print, rrr, profile) = ut.inject2(__name__, '[other.detecttrain]')


CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))


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
def classifier2_train(ibs, species_list=None, species_mapping={}, train_gid_set=None, **kwargs):
    from ibeis_cnn.ingest_ibeis import get_cnn_classifier2_training_images
    from ibeis_cnn.process import numpy_processed_directory3
    from ibeis_cnn.models.classifier2 import train_classifier2
    from ibeis_cnn.utils import save_model
    if species_list is not None:
        species_list = sorted(species_list)
    data_path = join(ibs.get_cachedir(), 'extracted')
    values = get_cnn_classifier2_training_images(ibs, species_list,
                                                 category_mapping=species_mapping,
                                                 train_gid_set=train_gid_set,
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
def aoi2_train(ibs, species_list=None, train_gid_list=None):
    from ibeis_cnn.ingest_ibeis import get_aoi2_training_data
    from ibeis_cnn.process import numpy_processed_directory5
    from ibeis_cnn.models.aoi2 import train_aoi2
    from ibeis_cnn.utils import save_model
    data_path = join(ibs.get_cachedir(), 'extracted')
    extracted_path = get_aoi2_training_data(ibs, dest_path=data_path,
                                            target_species_list=species_list,
                                            train_gid_list=train_gid_list)
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
        python -m ibeis.other.detecttrain
        python -m ibeis.other.detecttrain --allexamples
        python -m ibeis.other.detecttrain --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
