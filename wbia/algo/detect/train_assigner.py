# -*- coding: utf-8 -*-
# import logging
# from os.path import expanduser, join
# from wbia import constants as const
from wbia.control.controller_inject import (
    register_preprocs,
    # register_subprops,
    make_ibs_register_decorator,
)

from wbia.algo.detect.assigner import (
    gid_keyed_assigner_results,
    gid_keyed_ground_truth,
    illustrate_all_assignments,
    all_part_pairs,
)

import utool as ut
import numpy as np
from wbia import dtool
import random

# import os
from collections import OrderedDict

# from collections import defaultdict
from datetime import datetime
import time

from math import sqrt

from sklearn import preprocessing

# illustration imports
# from shutil import copy
# from PIL import Image, ImageDraw
# import wbia.plottool as pt


# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV


derived_attribute = register_preprocs['annot']


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


CLASSIFIER_OPTIONS = [
    {
        'name': 'Nearest Neighbors',
        'clf': KNeighborsClassifier(3),
        'param_options': {
            'n_neighbors': [3, 5, 11, 19],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
        },
    },
    {
        'name': 'Linear SVM',
        'clf': SVC(kernel='linear', C=0.025),
        'param_options': {
            'C': [1, 10, 100, 1000],
            'kernel': ['linear'],
        },
    },
    {
        'name': 'RBF SVM',
        'clf': SVC(gamma=2, C=1),
        'param_options': {
            'C': [1, 10, 100, 1000],
            'gamma': [0.001, 0.0001],
            'kernel': ['rbf'],
        },
    },
    {
        'name': 'Decision Tree',
        'clf': DecisionTreeClassifier(),  # max_depth=5
        'param_options': {
            'max_depth': np.arange(1, 12),
            'max_leaf_nodes': [2, 5, 10, 20, 50, 100],
        },
    },
    # {
    #     "name": "Random Forest",
    #     "clf": RandomForestClassifier(),  #max_depth=5, n_estimators=10, max_features=1
    #     "param_options": {
    #         'bootstrap': [True, False],
    #         'max_depth': [10, 50, 100, None],
    #         'max_features': ['auto', 'sqrt'],
    #         'min_samples_leaf': [1, 2, 4],
    #         'min_samples_split': [2, 5, 10],
    #         'n_estimators': [200, 1000, 2000]
    #     }
    # },
    # {
    #     "name": "Neural Net",
    #     "clf": MLPClassifier(),  #alpha=1, max_iter=1000
    #     "param_options": {
    #         'hidden_layer_sizes': [(10,30,10),(20,)],
    #         'activation': ['tanh', 'relu'],
    #         'solver': ['sgd', 'adam'],
    #         'alpha': [0.0001, 0.05],
    #         'learning_rate': ['constant','adaptive'],
    #     }
    # },
    {
        'name': 'AdaBoost',
        'clf': AdaBoostClassifier(),
        'param_options': {
            'n_estimators': np.arange(10, 310, 50),
            'learning_rate': [0.01, 0.05, 0.1, 1],
        },
    },
    {
        'name': 'Naive Bayes',
        'clf': GaussianNB(),
        'param_options': {},  # no hyperparams to optimize
    },
    {
        'name': 'QDA',
        'clf': QuadraticDiscriminantAnalysis(),
        'param_options': {'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]},
    },
]


classifier_names = [
    'Nearest Neighbors',
    'Linear SVM',
    'RBF SVM',
    'Decision Tree',
    'Random Forest',
    'Neural Net',
    'AdaBoost',
    'Naive Bayes',
    'QDA',
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel='linear', C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]


slow_classifier_names = 'Gaussian Process'
slow_classifiers = (GaussianProcessClassifier(1.0 * RBF(1.0)),)


def classifier_report(clf, name, assigner_data):
    print('%s CLASSIFIER REPORT ' % name)
    print('    %s: calling clf.fit' % str(datetime.now()))
    clf.fit(assigner_data['data'], assigner_data['target'])
    print('    %s: done training, making prediction ' % str(datetime.now()))
    preds = clf.predict(assigner_data['test'])
    print('    %s: done with predictions, computing accuracy' % str(datetime.now()))
    agree = [pred == truth for pred, truth in zip(preds, assigner_data['test_truth'])]
    accuracy = agree.count(True) / len(agree)
    print('    %s accuracy' % accuracy)
    print()
    return accuracy


@register_ibs_method
def compare_ass_classifiers(
    ibs, depc_table_name='assigner_viewpoint_features', print_accs=False
):

    assigner_data = ibs.wd_training_data(depc_table_name)

    accuracies = OrderedDict()
    for classifier in CLASSIFIER_OPTIONS:
        accuracy = classifier_report(classifier['clf'], classifier['name'], assigner_data)
        accuracies[classifier['name']] = accuracy

    # handy for e.g. pasting into excel
    if print_accs:
        just_accuracy = [accuracies[name] for name in accuracies.keys()]
        print(just_accuracy)

    return accuracies


@register_ibs_method
def tune_ass_classifiers(ibs, depc_table_name='assigner_viewpoint_unit_features'):

    assigner_data = ibs.wd_training_data(depc_table_name)

    accuracies = OrderedDict()
    best_acc = 0
    best_clf_name = ''
    best_clf_params = {}
    for classifier in CLASSIFIER_OPTIONS:
        print('Tuning %s' % classifier['name'])
        accuracy, best_params = ibs._tune_grid_search(
            classifier['clf'], classifier['param_options'], assigner_data
        )
        print()
        accuracies[classifier['name']] = {
            'accuracy': accuracy,
            'best_params': best_params,
        }
        if accuracy > best_acc:
            best_acc = accuracy
            best_clf_name = classifier['name']
            best_clf_params = best_params

    print(
        'best performance: %s using %s with params %s'
        % (best_acc, best_clf_name, best_clf_params)
    )

    return accuracies


@register_ibs_method
def _tune_grid_search(ibs, clf, parameters, assigner_data=None):
    if assigner_data is None:
        assigner_data = ibs.wd_training_data()

    X_train = assigner_data['data']
    y_train = assigner_data['target']
    X_test = assigner_data['test']
    y_test = assigner_data['test_truth']

    tune_search = GridSearchCV(  # TuneGridSearchCV(
        clf,
        parameters,
    )

    start = time.time()
    tune_search.fit(X_train, y_train)
    end = time.time()
    print('Tune Fit Time: %s' % (end - start))
    pred = tune_search.predict(X_test)
    accuracy = np.count_nonzero(np.array(pred) == np.array(y_test)) / len(pred)
    print('Tune Accuracy: %s' % accuracy)
    print('best parms   : %s' % tune_search.best_params_)

    return accuracy, tune_search.best_params_


@register_ibs_method
def _tune_random_search(ibs, clf, parameters, assigner_data=None):
    if assigner_data is None:
        assigner_data = ibs.wd_training_data()

    X_train = assigner_data['data']
    y_train = assigner_data['target']
    X_test = assigner_data['test']
    y_test = assigner_data['test_truth']

    tune_search = GridSearchCV(
        clf,
        parameters,
    )

    start = time.time()
    tune_search.fit(X_train, y_train)
    end = time.time()
    print('Tune Fit Time: %s' % (end - start))
    pred = tune_search.predict(X_test)
    accuracy = np.count_nonzero(np.array(pred) == np.array(y_test)) / len(pred)
    print('Tune Accuracy: %s' % accuracy)
    print('best parms   : %s' % tune_search.best_params_)

    return accuracy, tune_search.best_params_


# for wild dog dev
@register_ibs_method
def wd_assigner_data(ibs):
    return wd_training_data('part_assignment_features')


@register_ibs_method
def wd_normed_assigner_data(ibs):
    return wd_training_data('normalized_assignment_features')


@register_ibs_method
def wd_training_data(
    ibs, depc_table_name='assigner_viewpoint_features', balance_t_f=True
):
    all_aids = ibs.get_valid_aids()
    ia_classes = ibs.get_annot_species(all_aids)
    part_aids = [aid for aid, ia_class in zip(all_aids, ia_classes) if '+' in ia_class]
    part_gids = list(set(ibs.get_annot_gids(part_aids)))
    all_pairs = all_part_pairs(ibs, part_gids)
    all_feats = ibs.depc_annot.get(depc_table_name, all_pairs)
    names = [ibs.get_annot_names(all_pairs[0]), ibs.get_annot_names(all_pairs[1])]
    ground_truth = [n1 == n2 for (n1, n2) in zip(names[0], names[1])]

    # train_feats, test_feats = train_test_split(all_feats)
    # train_truth, test_truth = train_test_split(ground_truth)
    pairs_in_train = ibs.gid_train_test_split(
        all_pairs[0]
    )  # we could pass just the pair aids or just the body aids bc gids are the same
    train_feats, test_feats = split_list(all_feats, pairs_in_train)
    train_truth, test_truth = split_list(ground_truth, pairs_in_train)

    all_pairs_tuple = [(part, body) for part, body in zip(all_pairs[0], all_pairs[1])]
    train_pairs, test_pairs = split_list(all_pairs_tuple, pairs_in_train)

    if balance_t_f:
        train_balance_flags = balance_true_false_training_pairs(train_truth)
        train_truth = ut.compress(train_truth, train_balance_flags)
        train_feats = ut.compress(train_feats, train_balance_flags)
        train_pairs = ut.compress(train_pairs, train_balance_flags)

        test_balance_flags = balance_true_false_training_pairs(test_truth)
        test_truth = ut.compress(test_truth, test_balance_flags)
        test_feats = ut.compress(test_feats, test_balance_flags)
        test_pairs = ut.compress(test_pairs, test_balance_flags)

    assigner_data = {
        'data': train_feats,
        'target': train_truth,
        'test': test_feats,
        'test_truth': test_truth,
        'train_pairs': train_pairs,
        'test_pairs': test_pairs,
    }

    return assigner_data


# returns flags so we can compress other lists
def balance_true_false_training_pairs(ground_truth, seed=777):
    n_true = ground_truth.count(True)
    # there's always more false samples than true when we're looking at all pairs
    false_indices = [i for i, ground_t in enumerate(ground_truth) if not ground_t]
    import random

    random.seed(seed)
    subsampled_false_indices = random.sample(false_indices, n_true)
    # for quick membership check
    subsampled_false_indices = set(subsampled_false_indices)
    # keep all true flags, and the subsampled false ones
    keep_flags = [
        gt or (i in subsampled_false_indices) for i, gt in enumerate(ground_truth)
    ]
    return keep_flags


# def train_test_split(item_list, random_seed=777, test_size=0.1):
#     import random
#     import math

#     random.seed(random_seed)
#     sample_size = math.floor(len(item_list) * test_size)
#     all_indices = list(range(len(item_list)))
#     test_indices = random.sample(all_indices, sample_size)
#     test_items = [item_list[i] for i in test_indices]
#     train_indices = sorted(list(set(all_indices) - set(test_indices)))
#     train_items = [item_list[i] for i in train_indices]
#     return train_items, test_items


@register_ibs_method
def gid_train_test_split(ibs, aid_list, random_seed=777, test_size=0.1):
    r"""
    Makes a gid-wise train-test split. This avoids potential overfitting when a network
    is trained on some annots from one image and tested on others from the same image.

    Args:
        ibs         (IBEISController): IBEIS / WBIA controller object
        aid_list  (int): annot ids to split
        random_seed: to make this split reproducible
        test_size: portion of gids reserved for test data

    Yields:
        a boolean flag_list of which aids are in the training set. Returning the flag_list
        allows the user to filter multiple lists with one gid_train_test_split call


    CommandLine:
        python -m wbia.algo.detect.train_assigner gid_train_test_split

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> from wbia.algo.detect.assigner import *
        >>> from wbia.algo.detect.train_assigner import *
        >>> ibs = assigner_testdb_ibs()
        >>> aids = ibs.get_valid_aids()
        >>> all_gids = set(ibs.get_annot_gids(aids))
        >>> test_size = 0.34  # we want floor(test_size*3) to equal 1
        >>> aid_in_train = gid_train_test_split(ibs, aids, test_size=test_size)
        >>> train_aids = ut.compress(aids, aid_in_train)
        >>> aid_in_test = [not train for train in aid_in_train]
        >>> test_aids = ut.compress(aids, aid_in_test)
        >>> train_gids = set(ibs.get_annot_gids(train_aids))
        >>> test_gids = set(ibs.get_annot_gids(test_aids))
        >>> assert len(train_gids & test_gids) is 0
        >>> assert len(train_gids) + len(test_gids) == len(all_gids)
        >>> assert len(train_gids) is 2
        >>> assert len(test_gids) is 1
        >>> result = aid_in_train  # note one gid has 4 aids, the other two 2
        >>> print(result)
        [False, False, False, False, True, True, True, True]
    """
    print('calling gid_train_test_split')
    gid_list = ibs.get_annot_gids(aid_list)
    gid_set = list(set(gid_list))
    import math

    random.seed(random_seed)
    n_test_gids = math.floor(len(gid_set) * test_size)
    test_gids = set(random.sample(gid_set, n_test_gids))
    aid_in_train = [gid not in test_gids for gid in gid_list]
    return aid_in_train


def split_list(item_list, is_in_first_group_list):
    first_group = ut.compress(item_list, is_in_first_group_list)
    is_in_second_group = [not b for b in is_in_first_group_list]
    second_group = ut.compress(item_list, is_in_second_group)
    return first_group, second_group


def check_accuracy(ibs, assigner_data=None, cutoff_score=0.5, illustrate=False):

    if assigner_data is None:
        assigner_data = ibs.wd_training_data()

    all_aids = []
    for pair in assigner_data['test_pairs']:
        all_aids.extend(list(pair))
    all_aids = sorted(list(set(all_aids)))

    all_pairs, all_unassigned_aids = ibs.assign_parts(all_aids, cutoff_score)

    gid_to_assigner_results = gid_keyed_assigner_results(
        ibs, all_pairs, all_unassigned_aids
    )
    gid_to_ground_truth = gid_keyed_ground_truth(ibs, assigner_data)

    if illustrate:
        illustrate_all_assignments(ibs, gid_to_assigner_results, gid_to_ground_truth)

    correct_gids = []
    incorrect_gids = []
    gids_with_false_positives = 0
    n_false_positives = 0
    gids_with_false_negatives = 0
    gids_with_false_neg_allowing_errors = [0, 0, 0]
    max_allowed_errors = len(gids_with_false_neg_allowing_errors)
    n_false_negatives = 0
    gids_with_both_errors = 0
    for gid in gid_to_assigner_results.keys():
        assigned_pairs = set(gid_to_assigner_results[gid]['pairs'])
        ground_t_pairs = set(gid_to_ground_truth[gid]['pairs'])
        false_negatives = len(ground_t_pairs - assigned_pairs)
        false_positives = len(assigned_pairs - ground_t_pairs)
        n_false_negatives += false_negatives

        if false_negatives > 0:
            gids_with_false_negatives += 1
            if false_negatives >= 2:
                false_neg_log_index = min(
                    false_negatives - 2, max_allowed_errors - 1
                )  # ie, if we have 2 errors, we have a false neg even allowing 1 error, in index 0 of that list
                try:
                    gids_with_false_neg_allowing_errors[false_neg_log_index] += 1
                except Exception:
                    ut.embed()

        n_false_positives += false_positives
        if false_positives > 0:
            gids_with_false_positives += 1
        if false_negatives > 0 and false_positives > 0:
            gids_with_both_errors += 1

        pairs_equal = sorted(gid_to_assigner_results[gid]['pairs']) == sorted(
            gid_to_ground_truth[gid]['pairs']
        )
        if pairs_equal:
            correct_gids += [gid]
        else:
            incorrect_gids += [gid]

    n_gids = len(gid_to_assigner_results.keys())
    accuracy = len(correct_gids) / n_gids
    incorrect_gids = n_gids - len(correct_gids)
    acc_allowing_errors = [
        1 - (nerrors / n_gids) for nerrors in gids_with_false_neg_allowing_errors
    ]
    print('accuracy with cutoff of %s: %s' % (cutoff_score, accuracy))
    for i, acc_allowing_error in enumerate(acc_allowing_errors):
        print('        allowing %s errors, acc = %s' % (i + 1, acc_allowing_error))
    print(
        '        %s false positives on %s error images'
        % (n_false_positives, gids_with_false_positives)
    )
    print(
        '        %s false negatives on %s error images'
        % (n_false_negatives, gids_with_false_negatives)
    )
    print('        %s images with both errors' % (gids_with_both_errors))
    return accuracy


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.detect.train_assigner --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()


# additional assigner features to explore
class PartAssignmentFeatureConfig(dtool.Config):
    _param_info_list = []


@derived_attribute(
    tablename='part_assignment_features',
    parents=['annotations', 'annotations'],
    colnames=[
        'p_xtl',
        'p_ytl',
        'p_w',
        'p_h',
        'b_xtl',
        'b_ytl',
        'b_w',
        'b_h',
        'int_xtl',
        'int_ytl',
        'int_w',
        'int_h',
        'intersect_area_relative_part',
        'intersect_area_relative_body',
        'part_area_relative_body',
    ],
    coltypes=[
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        float,
        float,
        float,
    ],
    configclass=PartAssignmentFeatureConfig,
    fname='part_assignment_features',
    rm_extern_on_delete=True,
    chunksize=256,
)
def compute_assignment_features(depc, part_aid_list, body_aid_list, config=None):

    ibs = depc.controller

    part_gids = ibs.get_annot_gids(part_aid_list)
    body_gids = ibs.get_annot_gids(body_aid_list)
    assert (
        part_gids == body_gids
    ), 'can only compute assignment features on aids in the same image'
    parts_are_parts = ibs._are_part_annots(part_aid_list)
    assert all(parts_are_parts), 'all part_aids must be part annots.'
    bodies_are_parts = ibs._are_part_annots(body_aid_list)
    assert not any(bodies_are_parts), 'body_aids cannot be part annots'

    part_bboxes = ibs.get_annot_bboxes(part_aid_list)
    body_bboxes = ibs.get_annot_bboxes(body_aid_list)

    part_areas = [bbox[2] * bbox[3] for bbox in part_bboxes]
    body_areas = [bbox[2] * bbox[3] for bbox in body_bboxes]
    part_area_relative_body = [
        part_area / body_area for (part_area, body_area) in zip(part_areas, body_areas)
    ]

    intersect_bboxes = _bbox_intersections(part_bboxes, body_bboxes)
    # note that intesect w and h could be negative if there is no intersection, in which case it is the x/y distance between the annots.
    intersect_areas = [
        w * h if w > 0 and h > 0 else 0 for (_, _, w, h) in intersect_bboxes
    ]

    int_area_relative_part = [
        int_area / part_area for int_area, part_area in zip(intersect_areas, part_areas)
    ]
    int_area_relative_body = [
        int_area / body_area for int_area, body_area in zip(intersect_areas, body_areas)
    ]

    result_list = list(
        zip(
            part_bboxes,
            body_bboxes,
            intersect_bboxes,
            int_area_relative_part,
            int_area_relative_body,
            part_area_relative_body,
        )
    )

    for (
        part_bbox,
        body_bbox,
        intersect_bbox,
        int_area_relative_part,
        int_area_relative_body,
        part_area_relative_body,
    ) in result_list:
        yield (
            part_bbox[0],
            part_bbox[1],
            part_bbox[2],
            part_bbox[3],
            body_bbox[0],
            body_bbox[1],
            body_bbox[2],
            body_bbox[3],
            intersect_bbox[0],
            intersect_bbox[1],
            intersect_bbox[2],
            intersect_bbox[3],
            int_area_relative_part,
            int_area_relative_body,
            part_area_relative_body,
        )


@derived_attribute(
    tablename='normalized_assignment_features',
    parents=['annotations', 'annotations'],
    colnames=[
        'p_xtl',
        'p_ytl',
        'p_w',
        'p_h',
        'b_xtl',
        'b_ytl',
        'b_w',
        'b_h',
        'int_xtl',
        'int_ytl',
        'int_w',
        'int_h',
        'intersect_area_relative_part',
        'intersect_area_relative_body',
        'part_area_relative_body',
    ],
    coltypes=[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    configclass=PartAssignmentFeatureConfig,
    fname='normalized_assignment_features',
    rm_extern_on_delete=True,
    chunksize=256,
)
def normalized_assignment_features(depc, part_aid_list, body_aid_list, config=None):

    ibs = depc.controller

    part_gids = ibs.get_annot_gids(part_aid_list)
    body_gids = ibs.get_annot_gids(body_aid_list)
    assert (
        part_gids == body_gids
    ), 'can only compute assignment features on aids in the same image'
    parts_are_parts = ibs._are_part_annots(part_aid_list)
    assert all(parts_are_parts), 'all part_aids must be part annots.'
    bodies_are_parts = ibs._are_part_annots(body_aid_list)
    assert not any(bodies_are_parts), 'body_aids cannot be part annots'

    part_bboxes = ibs.get_annot_bboxes(part_aid_list)
    body_bboxes = ibs.get_annot_bboxes(body_aid_list)
    im_widths = ibs.get_image_widths(part_gids)
    im_heights = ibs.get_image_heights(part_gids)
    part_bboxes = _norm_bboxes(part_bboxes, im_widths, im_heights)
    body_bboxes = _norm_bboxes(body_bboxes, im_widths, im_heights)

    part_areas = [bbox[2] * bbox[3] for bbox in part_bboxes]
    body_areas = [bbox[2] * bbox[3] for bbox in body_bboxes]
    part_area_relative_body = [
        part_area / body_area for (part_area, body_area) in zip(part_areas, body_areas)
    ]

    intersect_bboxes = _bbox_intersections(part_bboxes, body_bboxes)
    # note that intesect w and h could be negative if there is no intersection, in which case it is the x/y distance between the annots.
    intersect_areas = [
        w * h if w > 0 and h > 0 else 0 for (_, _, w, h) in intersect_bboxes
    ]

    int_area_relative_part = [
        int_area / part_area for int_area, part_area in zip(intersect_areas, part_areas)
    ]
    int_area_relative_body = [
        int_area / body_area for int_area, body_area in zip(intersect_areas, body_areas)
    ]

    result_list = list(
        zip(
            part_bboxes,
            body_bboxes,
            intersect_bboxes,
            int_area_relative_part,
            int_area_relative_body,
            part_area_relative_body,
        )
    )

    for (
        part_bbox,
        body_bbox,
        intersect_bbox,
        int_area_relative_part,
        int_area_relative_body,
        part_area_relative_body,
    ) in result_list:
        yield (
            part_bbox[0],
            part_bbox[1],
            part_bbox[2],
            part_bbox[3],
            body_bbox[0],
            body_bbox[1],
            body_bbox[2],
            body_bbox[3],
            intersect_bbox[0],
            intersect_bbox[1],
            intersect_bbox[2],
            intersect_bbox[3],
            int_area_relative_part,
            int_area_relative_body,
            part_area_relative_body,
        )


@derived_attribute(
    tablename='standardized_assignment_features',
    parents=['annotations', 'annotations'],
    colnames=[
        'p_xtl',
        'p_ytl',
        'p_w',
        'p_h',
        'b_xtl',
        'b_ytl',
        'b_w',
        'b_h',
        'int_xtl',
        'int_ytl',
        'int_w',
        'int_h',
        'intersect_area_relative_part',
        'intersect_area_relative_body',
        'part_area_relative_body',
    ],
    coltypes=[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    configclass=PartAssignmentFeatureConfig,
    fname='standardized_assignment_features',
    rm_extern_on_delete=True,
    chunksize=256000000,  # chunk size is huge bc we need accurate means and stdevs of various traits
)
def standardized_assignment_features(depc, part_aid_list, body_aid_list, config=None):

    ibs = depc.controller

    part_gids = ibs.get_annot_gids(part_aid_list)
    body_gids = ibs.get_annot_gids(body_aid_list)
    assert (
        part_gids == body_gids
    ), 'can only compute assignment features on aids in the same image'
    parts_are_parts = ibs._are_part_annots(part_aid_list)
    assert all(parts_are_parts), 'all part_aids must be part annots.'
    bodies_are_parts = ibs._are_part_annots(body_aid_list)
    assert not any(bodies_are_parts), 'body_aids cannot be part annots'

    part_bboxes = ibs.get_annot_bboxes(part_aid_list)
    body_bboxes = ibs.get_annot_bboxes(body_aid_list)
    im_widths = ibs.get_image_widths(part_gids)
    im_heights = ibs.get_image_heights(part_gids)
    part_bboxes = _norm_bboxes(part_bboxes, im_widths, im_heights)
    body_bboxes = _norm_bboxes(body_bboxes, im_widths, im_heights)

    part_areas = [bbox[2] * bbox[3] for bbox in part_bboxes]
    body_areas = [bbox[2] * bbox[3] for bbox in body_bboxes]
    part_area_relative_body = [
        part_area / body_area for (part_area, body_area) in zip(part_areas, body_areas)
    ]

    intersect_bboxes = _bbox_intersections(part_bboxes, body_bboxes)
    # note that intesect w and h could be negative if there is no intersection, in which case it is the x/y distance between the annots.
    intersect_areas = [
        w * h if w > 0 and h > 0 else 0 for (_, _, w, h) in intersect_bboxes
    ]

    int_area_relative_part = [
        int_area / part_area for int_area, part_area in zip(intersect_areas, part_areas)
    ]
    int_area_relative_body = [
        int_area / body_area for int_area, body_area in zip(intersect_areas, body_areas)
    ]

    int_area_relative_part = preprocessing.scale(int_area_relative_part)
    int_area_relative_body = preprocessing.scale(int_area_relative_body)
    part_area_relative_body = preprocessing.scale(part_area_relative_body)

    result_list = list(
        zip(
            part_bboxes,
            body_bboxes,
            intersect_bboxes,
            int_area_relative_part,
            int_area_relative_body,
            part_area_relative_body,
        )
    )

    for (
        part_bbox,
        body_bbox,
        intersect_bbox,
        int_area_relative_part,
        int_area_relative_body,
        part_area_relative_body,
    ) in result_list:
        yield (
            part_bbox[0],
            part_bbox[1],
            part_bbox[2],
            part_bbox[3],
            body_bbox[0],
            body_bbox[1],
            body_bbox[2],
            body_bbox[3],
            intersect_bbox[0],
            intersect_bbox[1],
            intersect_bbox[2],
            intersect_bbox[3],
            int_area_relative_part,
            int_area_relative_body,
            part_area_relative_body,
        )


# like the above but bboxes are also standardized
@derived_attribute(
    tablename='mega_standardized_assignment_features',
    parents=['annotations', 'annotations'],
    colnames=[
        'p_xtl',
        'p_ytl',
        'p_w',
        'p_h',
        'b_xtl',
        'b_ytl',
        'b_w',
        'b_h',
        'int_xtl',
        'int_ytl',
        'int_w',
        'int_h',
        'intersect_area_relative_part',
        'intersect_area_relative_body',
        'part_area_relative_body',
    ],
    coltypes=[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    configclass=PartAssignmentFeatureConfig,
    fname='mega_standardized_assignment_features',
    rm_extern_on_delete=True,
    chunksize=256000000,  # chunk size is huge bc we need accurate means and stdevs of various traits
)
def mega_standardized_assignment_features(
    depc, part_aid_list, body_aid_list, config=None
):

    ibs = depc.controller

    part_gids = ibs.get_annot_gids(part_aid_list)
    body_gids = ibs.get_annot_gids(body_aid_list)
    assert (
        part_gids == body_gids
    ), 'can only compute assignment features on aids in the same image'
    parts_are_parts = ibs._are_part_annots(part_aid_list)
    assert all(parts_are_parts), 'all part_aids must be part annots.'
    bodies_are_parts = ibs._are_part_annots(body_aid_list)
    assert not any(bodies_are_parts), 'body_aids cannot be part annots'

    part_bboxes = ibs.get_annot_bboxes(part_aid_list)
    body_bboxes = ibs.get_annot_bboxes(body_aid_list)
    im_widths = ibs.get_image_widths(part_gids)
    im_heights = ibs.get_image_heights(part_gids)
    part_bboxes = _norm_bboxes(part_bboxes, im_widths, im_heights)
    body_bboxes = _norm_bboxes(body_bboxes, im_widths, im_heights)

    part_bboxes = _standardized_bboxes(part_bboxes)
    body_bboxes = _standardized_bboxes(body_bboxes)

    part_areas = [bbox[2] * bbox[3] for bbox in part_bboxes]
    body_areas = [bbox[2] * bbox[3] for bbox in body_bboxes]
    part_area_relative_body = [
        part_area / body_area for (part_area, body_area) in zip(part_areas, body_areas)
    ]

    intersect_bboxes = _bbox_intersections(part_bboxes, body_bboxes)
    # note that intesect w and h could be negative if there is no intersection, in which case it is the x/y distance between the annots.
    intersect_areas = [
        w * h if w > 0 and h > 0 else 0 for (_, _, w, h) in intersect_bboxes
    ]

    int_area_relative_part = [
        int_area / part_area for int_area, part_area in zip(intersect_areas, part_areas)
    ]
    int_area_relative_body = [
        int_area / body_area for int_area, body_area in zip(intersect_areas, body_areas)
    ]

    int_area_relative_part = preprocessing.scale(int_area_relative_part)
    int_area_relative_body = preprocessing.scale(int_area_relative_body)
    part_area_relative_body = preprocessing.scale(part_area_relative_body)

    result_list = list(
        zip(
            part_bboxes,
            body_bboxes,
            intersect_bboxes,
            int_area_relative_part,
            int_area_relative_body,
            part_area_relative_body,
        )
    )

    for (
        part_bbox,
        body_bbox,
        intersect_bbox,
        int_area_relative_part,
        int_area_relative_body,
        part_area_relative_body,
    ) in result_list:
        yield (
            part_bbox[0],
            part_bbox[1],
            part_bbox[2],
            part_bbox[3],
            body_bbox[0],
            body_bbox[1],
            body_bbox[2],
            body_bbox[3],
            intersect_bbox[0],
            intersect_bbox[1],
            intersect_bbox[2],
            intersect_bbox[3],
            int_area_relative_part,
            int_area_relative_body,
            part_area_relative_body,
        )


@derived_attribute(
    tablename='theta_assignment_features',
    parents=['annotations', 'annotations'],
    colnames=[
        'p_v1_x',
        'p_v1_y',
        'p_v2_x',
        'p_v2_y',
        'p_v3_x',
        'p_v3_y',
        'p_v4_x',
        'p_v4_y',
        'p_center_x',
        'p_center_y',
        'b_xtl',
        'b_ytl',
        'b_xbr',
        'b_ybr',
        'b_center_x',
        'b_center_y',
        'int_area_scalar',
        'part_body_distance',
        'part_body_centroid_dist',
        'int_over_union',
        'int_over_part',
        'int_over_body',
        'part_over_body',
    ],
    coltypes=[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    configclass=PartAssignmentFeatureConfig,
    fname='theta_assignment_features',
    rm_extern_on_delete=True,
    chunksize=256,  # chunk size is huge bc we need accurate means and stdevs of various traits
)
def theta_assignment_features(depc, part_aid_list, body_aid_list, config=None):

    from shapely import geometry
    import math

    ibs = depc.controller

    part_gids = ibs.get_annot_gids(part_aid_list)
    body_gids = ibs.get_annot_gids(body_aid_list)
    assert (
        part_gids == body_gids
    ), 'can only compute assignment features on aids in the same image'
    parts_are_parts = ibs._are_part_annots(part_aid_list)
    assert all(parts_are_parts), 'all part_aids must be part annots.'
    bodies_are_parts = ibs._are_part_annots(body_aid_list)
    assert not any(bodies_are_parts), 'body_aids cannot be part annots'

    im_widths = ibs.get_image_widths(part_gids)
    im_heights = ibs.get_image_heights(part_gids)

    part_verts = ibs.get_annot_rotated_verts(part_aid_list)
    body_verts = ibs.get_annot_rotated_verts(body_aid_list)
    part_verts = _norm_vertices(part_verts, im_widths, im_heights)
    body_verts = _norm_vertices(body_verts, im_widths, im_heights)
    part_polys = [geometry.Polygon(vert) for vert in part_verts]
    body_polys = [geometry.Polygon(vert) for vert in body_verts]
    intersect_polys = [
        part.intersection(body) for part, body in zip(part_polys, body_polys)
    ]
    intersect_areas = [poly.area for poly in intersect_polys]
    # just to make int_areas more comparable via ML methods, and since all distances < 1
    int_area_scalars = [math.sqrt(area) for area in intersect_areas]

    part_bboxes = ibs.get_annot_bboxes(part_aid_list)
    body_bboxes = ibs.get_annot_bboxes(body_aid_list)
    part_bboxes = _norm_bboxes(part_bboxes, im_widths, im_heights)
    body_bboxes = _norm_bboxes(body_bboxes, im_widths, im_heights)
    part_areas = [bbox[2] * bbox[3] for bbox in part_bboxes]
    body_areas = [bbox[2] * bbox[3] for bbox in body_bboxes]
    union_areas = [
        part + body - intersect
        for (part, body, intersect) in zip(part_areas, body_areas, intersect_areas)
    ]
    int_over_unions = [
        intersect / union for (intersect, union) in zip(intersect_areas, union_areas)
    ]

    part_body_distances = [
        part.distance(body) for part, body in zip(part_polys, body_polys)
    ]

    part_centroids = [poly.centroid for poly in part_polys]
    body_centroids = [poly.centroid for poly in body_polys]

    part_body_centroid_dists = [
        part.distance(body) for part, body in zip(part_centroids, body_centroids)
    ]

    int_over_parts = [
        int_area / part_area for part_area, int_area in zip(part_areas, intersect_areas)
    ]

    int_over_bodys = [
        int_area / body_area for body_area, int_area in zip(body_areas, intersect_areas)
    ]

    part_over_bodys = [
        part_area / body_area for part_area, body_area in zip(part_areas, body_areas)
    ]

    # note that here only parts have thetas, hence only returning body bboxes
    result_list = list(
        zip(
            part_verts,
            part_centroids,
            body_bboxes,
            body_centroids,
            int_area_scalars,
            part_body_distances,
            part_body_centroid_dists,
            int_over_unions,
            int_over_parts,
            int_over_bodys,
            part_over_bodys,
        )
    )

    for (
        part_vert,
        part_center,
        body_bbox,
        body_center,
        int_area_scalar,
        part_body_distance,
        part_body_centroid_dist,
        int_over_union,
        int_over_part,
        int_over_body,
        part_over_body,
    ) in result_list:
        yield (
            part_vert[0][0],
            part_vert[0][1],
            part_vert[1][0],
            part_vert[1][1],
            part_vert[2][0],
            part_vert[2][1],
            part_vert[3][0],
            part_vert[3][1],
            part_center.x,
            part_center.y,
            body_bbox[0],
            body_bbox[1],
            body_bbox[2],
            body_bbox[3],
            body_center.x,
            body_center.y,
            int_area_scalar,
            part_body_distance,
            part_body_centroid_dist,
            int_over_union,
            int_over_part,
            int_over_body,
            part_over_body,
        )


@derived_attribute(
    tablename='assigner_viewpoint_unit_features',
    parents=['annotations', 'annotations'],
    colnames=[
        'p_v1_x',
        'p_v1_y',
        'p_v2_x',
        'p_v2_y',
        'p_v3_x',
        'p_v3_y',
        'p_v4_x',
        'p_v4_y',
        'p_center_x',
        'p_center_y',
        'b_xtl',
        'b_ytl',
        'b_xbr',
        'b_ybr',
        'b_center_x',
        'b_center_y',
        'int_area_scalar',
        'part_body_distance',
        'part_body_centroid_dist',
        'int_over_union',
        'int_over_part',
        'int_over_body',
        'part_over_body',
        'part_is_left',
        'part_is_right',
        'part_is_up',
        'part_is_down',
        'part_is_front',
        'part_is_back',
        'body_is_left',
        'body_is_right',
        'body_is_up',
        'body_is_down',
        'body_is_front',
        'body_is_back',
    ],
    coltypes=[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    configclass=PartAssignmentFeatureConfig,
    fname='assigner_viewpoint_unit_features',
    rm_extern_on_delete=True,
    chunksize=256,  # chunk size is huge bc we need accurate means and stdevs of various traits
)
def assigner_viewpoint_unit_features(depc, part_aid_list, body_aid_list, config=None):

    from shapely import geometry
    import math

    ibs = depc.controller

    part_gids = ibs.get_annot_gids(part_aid_list)
    body_gids = ibs.get_annot_gids(body_aid_list)
    assert (
        part_gids == body_gids
    ), 'can only compute assignment features on aids in the same image'
    parts_are_parts = ibs._are_part_annots(part_aid_list)
    assert all(parts_are_parts), 'all part_aids must be part annots.'
    bodies_are_parts = ibs._are_part_annots(body_aid_list)
    assert not any(bodies_are_parts), 'body_aids cannot be part annots'

    im_widths = ibs.get_image_widths(part_gids)
    im_heights = ibs.get_image_heights(part_gids)

    part_verts = ibs.get_annot_rotated_verts(part_aid_list)
    body_verts = ibs.get_annot_rotated_verts(body_aid_list)
    part_verts = _norm_vertices(part_verts, im_widths, im_heights)
    body_verts = _norm_vertices(body_verts, im_widths, im_heights)
    part_polys = [geometry.Polygon(vert) for vert in part_verts]
    body_polys = [geometry.Polygon(vert) for vert in body_verts]
    intersect_polys = [
        part.intersection(body) for part, body in zip(part_polys, body_polys)
    ]
    intersect_areas = [poly.area for poly in intersect_polys]
    # just to make int_areas more comparable via ML methods, and since all distances < 1
    int_area_scalars = [math.sqrt(area) for area in intersect_areas]

    part_bboxes = ibs.get_annot_bboxes(part_aid_list)
    body_bboxes = ibs.get_annot_bboxes(body_aid_list)
    part_bboxes = _norm_bboxes(part_bboxes, im_widths, im_heights)
    body_bboxes = _norm_bboxes(body_bboxes, im_widths, im_heights)
    part_areas = [bbox[2] * bbox[3] for bbox in part_bboxes]
    body_areas = [bbox[2] * bbox[3] for bbox in body_bboxes]
    union_areas = [
        part + body - intersect
        for (part, body, intersect) in zip(part_areas, body_areas, intersect_areas)
    ]
    int_over_unions = [
        intersect / union for (intersect, union) in zip(intersect_areas, union_areas)
    ]

    part_body_distances = [
        part.distance(body) for part, body in zip(part_polys, body_polys)
    ]

    part_centroids = [poly.centroid for poly in part_polys]
    body_centroids = [poly.centroid for poly in body_polys]

    part_body_centroid_dists = [
        part.distance(body) for part, body in zip(part_centroids, body_centroids)
    ]

    int_over_parts = [
        int_area / part_area for part_area, int_area in zip(part_areas, intersect_areas)
    ]

    int_over_bodys = [
        int_area / body_area for body_area, int_area in zip(body_areas, intersect_areas)
    ]

    part_over_bodys = [
        part_area / body_area for part_area, body_area in zip(part_areas, body_areas)
    ]

    part_lrudfb_vects = get_annot_lrudfb_unit_vector(ibs, part_aid_list)
    body_lrudfb_vects = get_annot_lrudfb_unit_vector(ibs, part_aid_list)

    # note that here only parts have thetas, hence only returning body bboxes
    result_list = list(
        zip(
            part_verts,
            part_centroids,
            body_bboxes,
            body_centroids,
            int_area_scalars,
            part_body_distances,
            part_body_centroid_dists,
            int_over_unions,
            int_over_parts,
            int_over_bodys,
            part_over_bodys,
            part_lrudfb_vects,
            body_lrudfb_vects,
        )
    )

    for (
        part_vert,
        part_center,
        body_bbox,
        body_center,
        int_area_scalar,
        part_body_distance,
        part_body_centroid_dist,
        int_over_union,
        int_over_part,
        int_over_body,
        part_over_body,
        part_lrudfb_vect,
        body_lrudfb_vect,
    ) in result_list:
        ans = (
            part_vert[0][0],
            part_vert[0][1],
            part_vert[1][0],
            part_vert[1][1],
            part_vert[2][0],
            part_vert[2][1],
            part_vert[3][0],
            part_vert[3][1],
            part_center.x,
            part_center.y,
            body_bbox[0],
            body_bbox[1],
            body_bbox[2],
            body_bbox[3],
            body_center.x,
            body_center.y,
            int_area_scalar,
            part_body_distance,
            part_body_centroid_dist,
            int_over_union,
            int_over_part,
            int_over_body,
            part_over_body,
        )
        ans += tuple(part_lrudfb_vect)
        ans += tuple(body_lrudfb_vect)
        yield ans


@derived_attribute(
    tablename='theta_standardized_assignment_features',
    parents=['annotations', 'annotations'],
    colnames=[
        'p_v1_x',
        'p_v1_y',
        'p_v2_x',
        'p_v2_y',
        'p_v3_x',
        'p_v3_y',
        'p_v4_x',
        'p_v4_y',
        'p_center_x',
        'p_center_y',
        'b_xtl',
        'b_ytl',
        'b_xbr',
        'b_ybr',
        'b_center_x',
        'b_center_y',
        'int_area_scalar',
        'part_body_distance',
        'part_body_centroid_dist',
        'int_over_union',
        'int_over_part',
        'int_over_body',
        'part_over_body',
    ],
    coltypes=[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    configclass=PartAssignmentFeatureConfig,
    fname='theta_standardized_assignment_features',
    rm_extern_on_delete=True,
    chunksize=2560000,  # chunk size is huge bc we need accurate means and stdevs of various traits
)
def theta_standardized_assignment_features(
    depc, part_aid_list, body_aid_list, config=None
):

    from shapely import geometry
    import math

    ibs = depc.controller

    part_gids = ibs.get_annot_gids(part_aid_list)
    body_gids = ibs.get_annot_gids(body_aid_list)
    assert (
        part_gids == body_gids
    ), 'can only compute assignment features on aids in the same image'
    parts_are_parts = ibs._are_part_annots(part_aid_list)
    assert all(parts_are_parts), 'all part_aids must be part annots.'
    bodies_are_parts = ibs._are_part_annots(body_aid_list)
    assert not any(bodies_are_parts), 'body_aids cannot be part annots'

    im_widths = ibs.get_image_widths(part_gids)
    im_heights = ibs.get_image_heights(part_gids)

    part_verts = ibs.get_annot_rotated_verts(part_aid_list)
    body_verts = ibs.get_annot_rotated_verts(body_aid_list)
    part_verts = _norm_vertices(part_verts, im_widths, im_heights)
    body_verts = _norm_vertices(body_verts, im_widths, im_heights)
    part_polys = [geometry.Polygon(vert) for vert in part_verts]
    body_polys = [geometry.Polygon(vert) for vert in body_verts]
    intersect_polys = [
        part.intersection(body) for part, body in zip(part_polys, body_polys)
    ]
    intersect_areas = [poly.area for poly in intersect_polys]
    # just to make int_areas more comparable via ML methods, and since all distances < 1
    int_area_scalars = [math.sqrt(area) for area in intersect_areas]
    int_area_scalars = preprocessing.scale(int_area_scalars)

    part_bboxes = ibs.get_annot_bboxes(part_aid_list)
    body_bboxes = ibs.get_annot_bboxes(body_aid_list)
    part_bboxes = _norm_bboxes(part_bboxes, im_widths, im_heights)
    body_bboxes = _norm_bboxes(body_bboxes, im_widths, im_heights)
    part_areas = [bbox[2] * bbox[3] for bbox in part_bboxes]
    body_areas = [bbox[2] * bbox[3] for bbox in body_bboxes]
    union_areas = [
        part + body - intersect
        for (part, body, intersect) in zip(part_areas, body_areas, intersect_areas)
    ]
    int_over_unions = [
        intersect / union for (intersect, union) in zip(intersect_areas, union_areas)
    ]
    int_over_unions = preprocessing.scale(int_over_unions)

    part_body_distances = [
        part.distance(body) for part, body in zip(part_polys, body_polys)
    ]
    part_body_distances = preprocessing.scale(part_body_distances)

    part_centroids = [poly.centroid for poly in part_polys]
    body_centroids = [poly.centroid for poly in body_polys]

    part_body_centroid_dists = [
        part.distance(body) for part, body in zip(part_centroids, body_centroids)
    ]
    part_body_centroid_dists = preprocessing.scale(part_body_centroid_dists)

    int_over_parts = [
        int_area / part_area for part_area, int_area in zip(part_areas, intersect_areas)
    ]
    int_over_parts = preprocessing.scale(int_over_parts)

    int_over_bodys = [
        int_area / body_area for body_area, int_area in zip(body_areas, intersect_areas)
    ]
    int_over_bodys = preprocessing.scale(int_over_bodys)

    part_over_bodys = [
        part_area / body_area for part_area, body_area in zip(part_areas, body_areas)
    ]
    part_over_bodys = preprocessing.scale(part_over_bodys)

    # standardization

    # note that here only parts have thetas, hence only returning body bboxes
    result_list = list(
        zip(
            part_verts,
            part_centroids,
            body_bboxes,
            body_centroids,
            int_area_scalars,
            part_body_distances,
            part_body_centroid_dists,
            int_over_unions,
            int_over_parts,
            int_over_bodys,
            part_over_bodys,
        )
    )

    for (
        part_vert,
        part_center,
        body_bbox,
        body_center,
        int_area_scalar,
        part_body_distance,
        part_body_centroid_dist,
        int_over_union,
        int_over_part,
        int_over_body,
        part_over_body,
    ) in result_list:
        yield (
            part_vert[0][0],
            part_vert[0][1],
            part_vert[1][0],
            part_vert[1][1],
            part_vert[2][0],
            part_vert[2][1],
            part_vert[3][0],
            part_vert[3][1],
            part_center.x,
            part_center.y,
            body_bbox[0],
            body_bbox[1],
            body_bbox[2],
            body_bbox[3],
            body_center.x,
            body_center.y,
            int_area_scalar,
            part_body_distance,
            part_body_centroid_dist,
            int_over_union,
            int_over_part,
            int_over_body,
            part_over_body,
        )


def get_annot_lrudfb_unit_vector(ibs, aid_list):
    from wbia.core_annots import get_annot_lrudfb_bools

    bool_arrays = get_annot_lrudfb_bools(ibs, aid_list)
    float_arrays = [[float(b) for b in lrudfb] for lrudfb in bool_arrays]
    lrudfb_lengths = [sqrt(lrudfb.count(True)) for lrudfb in bool_arrays]
    # lying just to avoid division by zero errors
    lrudfb_lengths = [length if length != 0 else -1 for length in lrudfb_lengths]
    unit_float_array = [
        [f / length for f in lrudfb]
        for lrudfb, length in zip(float_arrays, lrudfb_lengths)
    ]

    return unit_float_array


def _norm_bboxes(bbox_list, width_list, height_list):
    normed_boxes = [
        (bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h)
        for (bbox, w, h) in zip(bbox_list, width_list, height_list)
    ]
    return normed_boxes


def _norm_vertices(verts_list, width_list, height_list):
    normed_verts = [
        [[x / w, y / h] for x, y in vert]
        for vert, w, h in zip(verts_list, width_list, height_list)
    ]
    return normed_verts


# does this even make any sense? let's find out experimentally
def _standardized_bboxes(bbox_list):
    xtls = preprocessing.scale([bbox[0] for bbox in bbox_list])
    ytls = preprocessing.scale([bbox[1] for bbox in bbox_list])
    wids = preprocessing.scale([bbox[2] for bbox in bbox_list])
    heis = preprocessing.scale([bbox[3] for bbox in bbox_list])
    standardized_bboxes = list(zip(xtls, ytls, wids, heis))
    return standardized_bboxes


def _bbox_intersections(bboxes_a, bboxes_b):
    corner_bboxes_a = _bbox_to_corner_format(bboxes_a)
    corner_bboxes_b = _bbox_to_corner_format(bboxes_b)

    intersect_xtls = [
        max(xtl_a, xtl_b)
        for ((xtl_a, _, _, _), (xtl_b, _, _, _)) in zip(corner_bboxes_a, corner_bboxes_b)
    ]

    intersect_ytls = [
        max(ytl_a, ytl_b)
        for ((_, ytl_a, _, _), (_, ytl_b, _, _)) in zip(corner_bboxes_a, corner_bboxes_b)
    ]

    intersect_xbrs = [
        min(xbr_a, xbr_b)
        for ((_, _, xbr_a, _), (_, _, xbr_b, _)) in zip(corner_bboxes_a, corner_bboxes_b)
    ]

    intersect_ybrs = [
        min(ybr_a, ybr_b)
        for ((_, _, _, ybr_a), (_, _, _, ybr_b)) in zip(corner_bboxes_a, corner_bboxes_b)
    ]

    intersect_widths = [
        int_xbr - int_xtl for int_xbr, int_xtl in zip(intersect_xbrs, intersect_xtls)
    ]

    intersect_heights = [
        int_ybr - int_ytl for int_ybr, int_ytl in zip(intersect_ybrs, intersect_ytls)
    ]

    intersect_bboxes = list(
        zip(intersect_xtls, intersect_ytls, intersect_widths, intersect_heights)
    )

    return intersect_bboxes


def _all_centroids(verts_list_a, verts_list_b):
    import shapely

    polys_a = [shapely.geometry.Polygon(vert) for vert in verts_list_a]
    polys_b = [shapely.geometry.Polygon(vert) for vert in verts_list_b]
    intersect_polys = [
        poly1.intersection(poly2) for poly1, poly2 in zip(polys_a, polys_b)
    ]

    centroids_a = [poly.centroid for poly in polys_a]
    centroids_b = [poly.centroid for poly in polys_b]
    centroids_int = [poly.centroid for poly in intersect_polys]

    return centroids_a, centroids_b, centroids_int


def _theta_aware_intersect_areas(verts_list_a, verts_list_b):
    import shapely

    polys_a = [shapely.geometry.Polygon(vert) for vert in verts_list_a]
    polys_b = [shapely.geometry.Polygon(vert) for vert in verts_list_b]
    intersect_areas = [
        poly1.intersection(poly2).area for poly1, poly2 in zip(polys_a, polys_b)
    ]
    return intersect_areas


# converts bboxes from (xtl, ytl, w, h) to (xtl, ytl, xbr, ybr)
def _bbox_to_corner_format(bboxes):
    corner_bboxes = [
        (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]) for bbox in bboxes
    ]
    return corner_bboxes


def _polygons_to_centroid_coords(polygon_list):
    centroids = [poly.centroid for poly in polygon_list]
    return centroids
