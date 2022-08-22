# -*- coding: utf-8 -*-
# import logging
# from os.path import expanduser, join
# from wbia import constants as const
import math
import random
import time

# import os
from collections import OrderedDict

# from collections import defaultdict
from datetime import datetime

import numpy as np
import utool as ut
from shapely import affinity, geometry
from sklearn import preprocessing
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from wbia import dtool
from wbia.algo.detect.assigner import (
    all_part_pairs,
    gid_keyed_assigner_results,
    gid_keyed_ground_truth,
    illustrate_all_assignments,
)
from wbia.control.controller_inject import (  # register_subprops,
    make_ibs_register_decorator,
    register_preprocs,
)
from wbia.core_annots import get_annot_lrudfb_unit_vector

# from math import sqrt


# illustration imports
# from shutil import copy
# from PIL import Image, ImageDraw
# import wbia.plottool as pt


derived_attribute = register_preprocs['annot']


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)

CURRENT_DEFAULT_FEATURE = 'assigner_viewpoint_unit_features'
FEATURE_OPTIONS = [
    'assigner_viewpoint_unit_features',
    'theta_standardized_assignment_features',
    'normalized_assignment_features',
    'theta_assignment_features',
    'assigner_viewpoint_features',
]

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
    # {
    #     'name': 'Linear SVM',
    #     'clf': SVC(kernel='linear', C=0.025),
    #     'param_options': {
    #         'C': [1, 10, 100, 1000],
    #         'kernel': ['linear'],
    #     },
    # },
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
    # {
    #     'name': 'QDA',
    #     'clf': QuadraticDiscriminantAnalysis(),
    #     'param_options': {'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]},
    # },
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
    ibs, depc_table_name=CURRENT_DEFAULT_FEATURE, print_accs=False
):

    assigner_data = turtle_training_data(ibs, depc_table_name)

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
def tune_ass_classifiers(ibs, depc_table_name=CURRENT_DEFAULT_FEATURE):

    assigner_data = turtle_training_data(ibs, depc_table_name)

    accuracies = OrderedDict()
    best_acc = 0
    best_clf_name = ''
    best_clf_params = {}
    for classifier in CLASSIFIER_OPTIONS:
        print('Tuning %s' % classifier['name'])
        accuracy, best_params = _tune_grid_search(
            ibs, classifier['clf'], classifier['param_options'], assigner_data
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


# works on the output of tune_ass_classifiers, prints in a way easy to copy-paste into a spreadsheet
def _print_tuning_results(ibs, accuracies=None, depc_table_name=CURRENT_DEFAULT_FEATURE):
    print()

    if accuracies is None:
        print("Tuning results with feature def'n %s (computing...)" % depc_table_name)
        accuracies = tune_ass_classifiers(ibs, depc_table_name)
        print()

    algo_names = list(accuracies.keys())
    for name in algo_names:
        print(name)
    accs = [accuracies[k]['accuracy'] for k in accuracies.keys()]
    print()
    for acc in accs:
        print(acc)
    args = [accuracies[k]['best_params'] for k in accuracies.keys()]
    print()
    for arg in args:
        print(arg)


def compare_tuned_features(ibs, feature_options=FEATURE_OPTIONS):
    for feat_name in feature_options:
        _print_tuning_results(ibs, depc_table_name=feat_name)


@register_ibs_method
def _tune_grid_search(ibs, clf, parameters, assigner_data=None):
    if assigner_data is None:
        assigner_data = turtle_training_data(ibs)

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
        assigner_data = turtle_training_data(ibs)

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
def wd_training_data(ibs, depc_table_name=CURRENT_DEFAULT_FEATURE, balance_t_f=True):
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
    pairs_in_train = gid_train_test_split(
        ibs, all_pairs[0]
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


@register_ibs_method
def turtle_training_data_iot(
    ibs, depc_table_name=CURRENT_DEFAULT_FEATURE, balance_t_f=False
):
    gids = ibs.get_valid_gids(imgsetid=1)  # already prepared and in this imgset
    all_aids = ut.flatten(ibs.get_image_aids(gids))

    ia_classes = ibs.get_annot_species(all_aids)
    part_aids = [aid for aid, ia_class in zip(all_aids, ia_classes) if '+' in ia_class]
    part_gids = list(set(ibs.get_annot_gids(part_aids)))
    all_pairs = all_part_pairs(ibs, part_gids)
    all_feats = ibs.depc_annot.get(depc_table_name, all_pairs)
    names = [ibs.get_annot_names(all_pairs[0]), ibs.get_annot_names(all_pairs[1])]
    ground_truth = [n1 == n2 for (n1, n2) in zip(names[0], names[1])]

    # train_feats, test_feats = train_test_split(all_feats)
    # train_truth, test_truth = train_test_split(ground_truth)
    pairs_in_train = gid_train_test_split(
        ibs, all_pairs[0]
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


# the assigner works on annots _only_ (since the detector does not create Parts atm) but some of our reference dbs have these saved as Part objects, not Annotations. This creates annotations for the parts
# WARNING: this will add names on unnamed aids that have a part (this is needed for the assigner to use them as a ground truth) so only run this on database you are willing to modify as such.
def save_parts_as_annots(ibs, part_rowids=None, part_name='head'):
    if part_rowids is None:
        part_rowids = ibs.get_valid_part_rowids()

    part_bboxes = ibs.get_part_bboxes(part_rowids)
    gids = ibs.get_part_image_rowids(part_rowids)
    # check for duplicates
    # part_has_no_annot_yet = [True] * len(part_rowids)
    # if not force:
    #     existing_aids = ibs.get_image_aids(gids)
    #     existing_bboxes = set([ibs.get_annot_bboxes(aids) for aids in existing_aids])
    #     part_has_no_annot_yet = [p_bbox not in existing_boxes for p_bbox, existing_boxes
    #                               in zip(part_bboxes, existing_bboxes)]
    #     n_duplicates = part_has_no_annot_yet.count(False)
    #     n_new = part_has_no_annot_yet.count(True)
    #     if (n_duplicates > 0):
    #         print("WARNING: found %s parts that already have annots in save_parts_as_annots. Skipping these and importing the %s new annots" % (n_duplicates, n_new))
    # part_rowids = ut.compress(part_rowids, part_has_no_annot_yet)
    # part_bboxes = ut.compress(part_bboxes, part_has_no_annot_yet)
    # gids = ut.compress(gids, part_has_no_annot_yet)

    parent_aids = ibs.get_part_aids(part_rowids)
    parent_species = ibs.get_annot_species(parent_aids)
    part_species = [species + '+' + part_name for species in parent_species]
    parent_names = ibs.get_annot_names(parent_aids)

    # for unnamed parents that *do* have parts, we need to add a unique (per image) name to both aids bc the parent/child link is seen by the assigner via them both having the same name
    unnamed_parents = [
        aid for aid, name in zip(parent_aids, parent_names) if name == '____'
    ]
    unnamed_parents_gids = ibs.get_annot_gids(unnamed_parents)
    unnamed_parents_image_indices = _get_image_indices(ibs, unnamed_parents)
    names_for_unnamed = [
        str(gid) + '-' + str(i) + '-assignerpair'
        for gid, i in zip(unnamed_parents_gids, unnamed_parents_image_indices)
    ]
    ibs.set_annot_names(unnamed_parents, names_for_unnamed)
    parent_names = ibs.get_annot_names(parent_aids)

    parent_nids = ibs.get_annot_name_rowids(parent_aids)

    thetas = ibs.get_part_thetas(part_rowids)
    viewps = ibs.get_part_viewpoints(part_rowids)
    qualts = ibs.get_part_qualities(part_rowids)
    confs = ibs.get_part_detect_confidence(part_rowids)
    notes = ibs.get_part_notes(part_rowids)

    new_aids = ibs.add_annots(
        gids,
        bbox_list=part_bboxes,
        theta_list=thetas,
        species_list=part_species,
        nid_list=parent_nids,
        viewpoint_list=viewps,
        quality_list=qualts,
        detect_confidence_list=confs,
        notes_list=notes,
    )
    print('save_parts_as_annots created %s new annotations from parts.' % len(new_aids))
    return new_aids


# developed for iot, this adds one false part/body pair to false_pair_per_image * num input images.
def create_dummy_assigner_aids(
    ibs,
    aid_list,
    fake_pairs_per_image=0.5,
    stop_after=math.inf,
    shrinkage=0.7,
    illustrate=True,
):
    gids = list(set(ibs.get_annot_gids(aid_list)))
    old_parts, old_bodies = all_part_pairs(ibs, gids)
    pair_gids = ibs.get_annot_gids(old_parts)

    # now construct transformation values for all the fakes
    random.seed(777)
    n_pairs = min(stop_after, math.floor(fake_pairs_per_image * len(gids)))
    # we choose which gids/pairs to sample from, and which we will paste into, without replacement.
    from_and_to_indices = random.sample(list(range(len(old_parts))), n_pairs * 2)
    pair_from_indices = from_and_to_indices[:n_pairs]
    copy_to_indices = from_and_to_indices[n_pairs:]

    source_parts = [old_parts[i] for i in pair_from_indices]
    source_bodies = [old_bodies[i] for i in pair_from_indices]
    source_gids = [pair_gids[i] for i in pair_from_indices]
    source_imwidths = ibs.get_image_widths(source_gids)
    source_imheights = ibs.get_image_heights(source_gids)
    source_part_verts = ibs.get_annot_verts(source_parts)
    source_body_verts = ibs.get_annot_verts(source_bodies)

    # get input args for make_verts_not_aoi
    shrinkage_sigma = (
        1 - shrinkage
    ) / 2  # heuristic such that >97% of the time, dummy bboxes have shrunk somewhat
    shrinkages = [random.gauss(shrinkage, shrinkage_sigma) for _ in range(n_pairs)]
    thetas = [random.gauss(0, 7) for _ in range(n_pairs)]  # so +/-15% is roughly 2 sigma
    perimeterizes = [0.6 for _ in range(n_pairs)]

    transform_input = zip(
        source_part_verts,
        source_body_verts,
        source_imwidths,
        source_imheights,
        shrinkages,
        thetas,
        perimeterizes,
    )

    transormed_verts = [
        make_verts_not_aoi(v1, v2, imwidth, imheight, shrink, theta, perimeterize)
        for (v1, v2, imwidth, imheight, shrink, theta, perimeterize) in transform_input
    ]

    # transform verts to new images
    target_gids = [pair_gids[i] for i in copy_to_indices]
    target_imwidths = ibs.get_image_widths(target_gids)
    target_imheights = ibs.get_image_heights(target_gids)
    transformed_verts = [
        _scale_verts_to_new_image(verts, imwidth, imheight, new_width, new_height)
        for verts, imwidth, imheight, new_width, new_height in zip(
            transormed_verts,
            source_imwidths,
            source_imheights,
            target_imwidths,
            target_imheights,
        )
    ]
    new_part_verts = [vertses[0] for vertses in transformed_verts]
    new_body_verts = [vertses[1] for vertses in transformed_verts]

    # save new annots with these verts on new images with a note on their origin
    source_part_species = ibs.get_annot_species(source_parts)
    source_body_species = ibs.get_annot_species(source_bodies)
    source_part_viewpoints = ibs.get_annot_viewpoints(source_parts)
    source_body_viewpoints = ibs.get_annot_viewpoints(source_bodies)
    # clean viewpoints
    source_part_viewpoints = [
        view if view != 'IGNORE' else None for view in source_part_viewpoints
    ]
    source_body_viewpoints = [
        view if view != 'IGNORE' else None for view in source_body_viewpoints
    ]

    source_names = ibs.get_annot_names(source_bodies)
    new_part_notes = [
        'dummy assigner aid transformed from part annot %s' % aid for aid in source_parts
    ]
    new_body_notes = [
        'dummy assigner aid transformed from body annot %s' % aid for aid in source_bodies
    ]

    dummy_parts = ibs.add_annots(
        target_gids,
        species_list=source_part_species,
        name_list=source_names,
        vert_list=new_part_verts,
        viewpoint_list=source_part_viewpoints,
        notes_list=new_part_notes,
    )

    dummy_bodies = ibs.add_annots(
        target_gids,
        species_list=source_body_species,
        name_list=source_names,
        vert_list=new_body_verts,
        viewpoint_list=source_body_viewpoints,
        notes_list=new_body_notes,
    )

    if illustrate:
        # reuse assigner illustrations
        ut.embed()
        gids_to_assigner_gtruth = _true_assignment_pair_dicts(ibs, gids)
        target_dir = '/tmp/dummies/'
        illustrate_all_assignments(
            ibs, gids_to_assigner_gtruth, gids_to_assigner_gtruth, target_dir, limit=50
        )

    return (dummy_parts, dummy_bodies)


# just used for illustration code above
def _true_assignment_pair_dicts(ibs, gids):
    gid_aids = ibs.get_image_aids(gids)
    gid_names = [ibs.get_annot_names(aids) for aids in gid_aids]
    gid_to_assigner_results = {}
    for gid, aids, names in zip(gids, gid_aids, gid_names):
        this_dict = {'pairs': [], 'unassigned': []}
        for name in names:
            pair = tuple(aid for aid, _name in zip(aids, names) if _name == name)
            this_dict['pairs'] += [pair]
        gid_to_assigner_results[gid] = this_dict

    return gid_to_assigner_results


# vertses is a tuple of *two* verts tuples (one for part, one for body)
def _scale_verts_to_new_image(
    vertses, old_imwidth, old_imheight, new_imwidth, new_imheight
):
    verts0 = vertses[0]
    verts1 = vertses[1]
    normalized_verts0 = [(x / old_imwidth, y / old_imheight) for (x, y) in verts0]
    normalized_verts1 = [(x / old_imwidth, y / old_imheight) for (x, y) in verts1]
    projected_verts0 = [
        (x * new_imwidth, y * new_imheight) for (x, y) in normalized_verts0
    ]
    projected_verts1 = [
        (x * new_imwidth, y * new_imheight) for (x, y) in normalized_verts1
    ]

    final_verts0 = [(math.floor(x), math.floor(y)) for (x, y) in projected_verts0]
    final_verts1 = [(math.floor(x), math.floor(y)) for (x, y) in projected_verts1]

    return (final_verts0, final_verts1)


def make_verts_not_aoi(
    verts1, verts2, imwidth, imheight, shrinkage=0.8, theta=0.0, perimeterize=0.5
):
    r"""
    Makes two dummy, non-aoi bboxes out of two real (presumed aoi) bboxes,
    preserving their relative geometry

    Args:
        verts1: annot verts to use as a reference
        verts2: annot verts to use as a reference
        imwidth: original image width
        imheight: original image height
        mirror: whether we mirror the verts along the y axis (L-R flipping)
        shrinkage: how much to shrink the box, assuming non-aoi annots are
            further away and thus smaller
        theta: how much to rotate the boxes (degrees)
        perimeterize: the boxes will be moved closer to the edge of the image by
            a factor of perimeterize thusly: after mirroring, shrinking, rotating,
            take the vertex Vx that is closest to the edge of the image. Translate
            all vertices equally such that the distance from Vx to that edge is
            reduced by a factor of perimeterize.

    Yields:
        transformed verts
    """
    poly1 = geometry.Polygon(verts1)
    poly2 = geometry.Polygon(verts2)

    # these if conditions are as much about code organization as anything
    # I don't feel like mirroring viewpoints and dont think mirroring adds value
    # if mirror:
    #     poly1 = affinity.scale(poly1, xfact=-1.0)
    #     poly2 = affinity.scale(poly2, xfact=-1.0)

    if shrinkage != 1.0:
        # shrink bboxes about the origin of their shared centroid
        centroid_both = _centroids_centroid(poly1, poly2)
        poly1 = affinity.scale(
            poly1, xfact=shrinkage, yfact=shrinkage, origin=centroid_both
        )
        poly2 = affinity.scale(
            poly2, xfact=shrinkage, yfact=shrinkage, origin=centroid_both
        )

    if theta != 0.0:
        centroid_both = _centroids_centroid(poly1, poly2)
        poly1 = affinity.rotate(poly1, theta, origin=centroid_both)
        poly2 = affinity.rotate(poly2, theta, origin=centroid_both)

    if perimeterize != 1.0:
        # distance to nearest edge and a tuple indicating its direction from the polys
        edge_distance, nearest_edge_direction = _nearest_edge_distance(
            poly1, poly2, imwidth, imheight
        )
        translate_distance = edge_distance * perimeterize
        x_offset = translate_distance * nearest_edge_direction[0]
        y_offset = translate_distance * nearest_edge_direction[1]
        poly1 = affinity.translate(poly1, x_offset, y_offset)
        poly2 = affinity.translate(poly2, x_offset, y_offset)

    new_verts1 = list(poly1.exterior.coords)[
        :4
    ]  # the coords list duplicates the last vertex
    new_verts2 = list(poly2.exterior.coords)[:4]

    return new_verts1, new_verts2


def _centroids_centroid(poly1, poly2):
    centroid1 = poly1.centroid
    centroid2 = poly2.centroid
    center_line = geometry.LineString((centroid1, centroid2))
    centroid_both = center_line.centroid
    return centroid_both


# returns both the distance and a tuple indicating which edge. eg (1,0) indicates the +x
# direction edge is nearist, (-1,0) would be the x origin edge.
def _nearest_edge_distance(poly1, poly2, imwidth, imheight):
    image_boundary_lines = [
        geometry.LineString(((0, 0), (imwidth, 0))),
        geometry.LineString(((imwidth, 0), (imwidth, imheight))),
        geometry.LineString(((imwidth, imheight), (0, imheight))),
        geometry.LineString(((0, imheight), (0, 0))),
    ]

    p1_im_boundary_dists = [poly1.distance(line) for line in image_boundary_lines]
    p2_im_boundary_dists = [poly1.distance(line) for line in image_boundary_lines]

    p1_dist = min(p1_im_boundary_dists)
    p2_dist = min(p2_im_boundary_dists)
    edge_distance = min(p1_dist, p2_dist)

    if p1_dist < p2_dist:
        # outermost_poly = poly1
        outermost_poly_dists = p1_im_boundary_dists
    else:
        # outermost_poly = poly2
        outermost_poly_dists = p2_im_boundary_dists

    # now figure out nearest_edge_direction, which is a directional vector
    nearest_edge = outermost_poly_dists.index(edge_distance)
    if nearest_edge == 0:
        # this edge is at y=0, so we move the poly closer by decresing y and keeping x constant
        nearest_edge_direction = (0, -1)
    elif nearest_edge == 1:
        # this edge is at x=max, so we move the poly closer by increasing x
        nearest_edge_direction = (1, 0)
    elif nearest_edge == 2:
        # this edge is at y=max
        nearest_edge_direction = (0, 1)
    else:
        nearest_edge_direction = (0, -1)

    return (edge_distance, nearest_edge_direction)


# gives each aid a number (0,1,2...) indicating which aid it is in its parent image
def _get_image_indices(ibs, aid_list):
    gids = ibs.get_annot_gids(aid_list)
    gids_aids = ibs.get_image_aids(gids)
    aid_indices = [gid_aids.index(aid) for (aid, gid_aids) in zip(aid_list, gids_aids)]
    return aid_indices


def turtle_training_data(ibs, depc_table_name=CURRENT_DEFAULT_FEATURE, balance_t_f=True):

    all_aids = ibs.get_valid_aids()
    all_species = ibs.get_annot_species(all_aids)
    is_turtle = ['turtle' in spec for spec in all_species]
    all_aids = ut.compress(all_aids, is_turtle)

    ia_classes = ibs.get_annot_species(all_aids)
    part_aids = [aid for aid, ia_class in zip(all_aids, ia_classes) if '+' in ia_class]
    part_gids = list(set(ibs.get_annot_gids(part_aids)))
    all_pairs = all_part_pairs(ibs, part_gids)
    names = [ibs.get_annot_names(all_pairs[0]), ibs.get_annot_names(all_pairs[1])]
    ground_truth = [n1 == n2 for (n1, n2) in zip(names[0], names[1])]

    if balance_t_f:
        keep_flags = balance_true_false_training_pairs(ground_truth)
        # janky bc all_pairs is a tuple so we can't do item-assignment
        all_pairs0 = ut.compress(all_pairs[0], keep_flags)
        all_pairs1 = ut.compress(all_pairs[1], keep_flags)
        all_pairs = (all_pairs0, all_pairs1)
        ground_truth = ut.compress(ground_truth, keep_flags)

    all_feats = ibs.depc_annot.get(depc_table_name, all_pairs)
    pairs_in_train = gid_train_test_split(
        ibs, all_pairs[0], test_size=0.3
    )  # we could pass just the pair aids or just the body aids bc gids are the same

    train_feats, test_feats = split_list(all_feats, pairs_in_train)
    train_truth, test_truth = split_list(ground_truth, pairs_in_train)

    all_pairs_tuple = [(part, body) for part, body in zip(all_pairs[0], all_pairs[1])]
    train_pairs, test_pairs = split_list(all_pairs_tuple, pairs_in_train)

    assigner_data = {
        'data': train_feats,
        'target': train_truth,
        'test': test_feats,
        'test_truth': test_truth,
        'train_pairs': train_pairs,
        'test_pairs': test_pairs,
    }

    return assigner_data


def _is_good_training_image(ibs, gid):
    aids = ibs.get_image_aids(gid)
    names = ibs.get_annot_names(aids)

    ia_classes = ibs.get_annot_species(aids)

    enough_aids = len(aids) > 2
    # if we have 2 or more '____' names we don't have a ground truth of which annot is which individual
    enough_names = len(set(names)) > 1 and names.count('____') <= 1
    # since species encodes iaclass; we need multiple ia classes in order to make any assignments
    enough_ia_classes = len(set(ia_classes)) > 1
    has_a_part = any(['+' in ia_class for ia_class in ia_classes])

    return enough_aids and enough_names and enough_ia_classes and has_a_part


# returns flags so we can compress other lists
def balance_true_false_training_pairs(ground_truth, seed=777):
    import random

    random.seed(seed)

    n_true = ground_truth.count(True)
    n_false = ground_truth.count(False)
    if n_false > n_true:
        false_indices = [i for i, ground_t in enumerate(ground_truth) if not ground_t]

        subsampled_false_indices = random.sample(false_indices, n_true)
        subsampled_false_indices = set(subsampled_false_indices)
        # keep all true indices, and the subsampled false ones
        keep_flags = [
            gt or (i in subsampled_false_indices) for i, gt in enumerate(ground_truth)
        ]
    else:
        true_indices = [i for i, ground_t in enumerate(ground_truth) if ground_t]

        subsampled_true_indices = random.sample(true_indices, n_false)
        # for quick membership check
        subsampled_true_indices = set(subsampled_true_indices)
        # keep all false indices, and the subsampled true ones
        keep_flags = [
            (not gt) or (i in subsampled_true_indices)
            for i, gt in enumerate(ground_truth)
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


def check_accuracy(
    ibs,
    assigner_data=None,
    cutoff_score=0.5,
    illustrate=False,
    only_illustrate_false=False,
    limit=20,
):

    if assigner_data is None:
        assigner_data = turtle_training_data(ibs)

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
        illustrate_all_assignments(
            ibs,
            gid_to_assigner_results,
            gid_to_ground_truth,
            only_false=only_illustrate_false,
            limit=limit,
        )

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
    print('accuracy with cutoff of {}: {}'.format(cutoff_score, accuracy))
    for i, acc_allowing_error in enumerate(acc_allowing_errors):
        print('        allowing {} errors, acc = {}'.format(i + 1, acc_allowing_error))
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

    import math

    from shapely import geometry

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
    tablename='old_assigner_old_viewpoint_unit_features',
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
    fname='old_assigner_old_viewpoint_unit_features',
    rm_extern_on_delete=True,
    chunksize=256,  # chunk size is huge bc we need accurate means and stdevs of various traits
)
def old_assigner_old_viewpoint_unit_features(
    depc, part_aid_list, body_aid_list, config=None
):

    import math

    from shapely import geometry

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

    import math

    from shapely import geometry

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
