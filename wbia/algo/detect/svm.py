# -*- coding: utf-8 -*-
"""
Interface to Darknet object proposals.
"""
from __future__ import absolute_import, division, print_function
import utool as ut
from os import listdir
from os.path import join, isfile, isdir

(print, rrr, profile) = ut.inject2(__name__, '[svm]')


VERBOSE_SVM = ut.get_argflag('--verbsvm') or ut.VERBOSE


CONFIG_URL_DICT = {
    # 'localizer-zebra-10'  : 'https://wildbookiarepository.azureedge.net/models/classifier.svm.localization.zebra.10.zip',
    # 'localizer-zebra-20'  : 'https://wildbookiarepository.azureedge.net/models/classifier.svm.localization.zebra.20.zip',
    # 'localizer-zebra-30'  : 'https://wildbookiarepository.azureedge.net/models/classifier.svm.localization.zebra.30.zip',
    # 'localizer-zebra-40'  : 'https://wildbookiarepository.azureedge.net/models/classifier.svm.localization.zebra.40.zip',
    # 'localizer-zebra-50'  : 'https://wildbookiarepository.azureedge.net/models/classifier.svm.localization.zebra.50.zip',
    # 'localizer-zebra-60'  : 'https://wildbookiarepository.azureedge.net/models/classifier.svm.localization.zebra.60.zip',
    # 'localizer-zebra-70'  : 'https://wildbookiarepository.azureedge.net/models/classifier.svm.localization.zebra.70.zip',
    # 'localizer-zebra-80'  : 'https://wildbookiarepository.azureedge.net/models/classifier.svm.localization.zebra.80.zip',
    # 'localizer-zebra-90'  : 'https://wildbookiarepository.azureedge.net/models/classifier.svm.localization.zebra.90.zip',
    # 'localizer-zebra-100' : 'https://wildbookiarepository.azureedge.net/models/classifier.svm.localization.zebra.100.zip',
    # 'image-zebra'         : 'https://wildbookiarepository.azureedge.net/models/classifier.svm.image.zebra.pkl',
    # 'default'             : 'https://wildbookiarepository.azureedge.net/models/classifier.svm.image.zebra.pkl',
    # None                  : 'https://wildbookiarepository.azureedge.net/models/classifier.svm.image.zebra.pkl',
}


def classify_helper(weight_filepath, vector_list, index_list=None, verbose=VERBOSE_SVM):
    if index_list is None:
        index_list = list(range(len(vector_list)))
    # Init score and class holders
    score_dict = {index: [] for index in index_list}
    class_dict = {index: [] for index in index_list}
    # Load models
    model_tup = ut.load_cPkl(weight_filepath, verbose=verbose)
    model, scaler = model_tup
    # Normalize
    vector_list = scaler.transform(vector_list)
    # calculate decisions and predictions
    # score_list = model.decision_function(vector_list)
    score_list = model.predict_proba(vector_list)
    # Take only the positive probability
    score_list = score_list[:, 1]
    class_list = model.predict(vector_list)
    # Zip together results
    zipped = zip(index_list, score_list, class_list)
    for index, score_, class_ in zipped:
        score_dict[index].append(score_)
        class_dict[index].append(class_)
    # Return scores and classes
    return score_dict, class_dict


def classify(vector_list, weight_filepath, verbose=VERBOSE_SVM, **kwargs):
    """
    Args:
        thumbail_list (list of str): the list of image thumbnails that need classifying

    Returns:
        iter
    """
    import multiprocessing
    import numpy as np

    # Get correct weight if specified with shorthand
    if weight_filepath in CONFIG_URL_DICT:
        weight_url = CONFIG_URL_DICT[weight_filepath]
        if weight_url.endswith('.zip'):
            weight_filepath = ut.grab_zipped_url(weight_url, appname='wbia')
        else:
            weight_filepath = ut.grab_file_url(
                weight_url, appname='wbia', check_hash=True
            )

    # Get ensemble
    is_ensemble = isdir(weight_filepath)
    if is_ensemble:
        weight_filepath_list = sorted(
            [
                join(weight_filepath, filename)
                for filename in listdir(weight_filepath)
                if isfile(join(weight_filepath, filename))
            ]
        )
    else:
        weight_filepath_list = [weight_filepath]
    num_weights = len(weight_filepath_list)
    assert num_weights > 0

    # Form dictionaries
    num_vectors = len(vector_list)
    index_list = list(range(num_vectors))

    # Generate parallelized wrapper
    OLD = False
    if is_ensemble and OLD:
        vectors_list = [vector_list for _ in range(num_weights)]
        args_list = zip(weight_filepath_list, vectors_list)
        nTasks = num_weights
        print('Processing ensembles in parallel using %d ensembles' % (num_weights,))
    else:
        num_cpus = multiprocessing.cpu_count()
        vector_batch = int(np.ceil(float(num_vectors) / num_cpus))
        vector_rounds = int(np.ceil(float(num_vectors) / vector_batch))

        args_list = []
        for vector_round in range(vector_rounds):
            start_index = vector_round * vector_batch
            stop_index = (vector_round + 1) * vector_batch
            assert start_index < num_vectors
            stop_index = min(stop_index, num_vectors)
            # print('Slicing index range: [%r, %r)' % (start_index, stop_index, ))

            # Slice gids and get feature data
            index_list_ = list(range(start_index, stop_index))
            vector_list_ = vector_list[start_index:stop_index]
            assert len(index_list_) == len(vector_list_)
            for weight_filepath in weight_filepath_list:
                args = (weight_filepath, vector_list_, index_list_)
                args_list.append(args)

        nTasks = len(args_list)
        print('Processing vectors in parallel using vector_batch = %r' % (vector_batch,))

    # Perform inference
    classify_iter = ut.generate2(
        classify_helper, args_list, nTasks=nTasks, ordered=True, force_serial=False
    )

    # Classify with SVM for each image vector
    score_dict = {index: [] for index in index_list}
    class_dict = {index: [] for index in index_list}
    for score_dict_, class_dict_ in classify_iter:
        for index in index_list:
            if index in score_dict_:
                score_dict[index] += score_dict_[index]
            if index in class_dict_:
                class_dict[index] += class_dict_[index]

    # Organize and compute mode and average for class and score
    for index in index_list:
        score_list_ = score_dict[index]
        class_list_ = class_dict[index]
        score_ = sum(score_list_) / len(score_list_)
        class_ = max(set(class_list_), key=class_list_.count)
        class_ = 'positive' if int(class_) == 1 else 'negative'
        yield score_, class_
