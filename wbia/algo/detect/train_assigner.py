import logging
from os.path import expanduser, join
from wbia import constants as const
from wbia.control.controller_inject import register_preprocs, register_subprops, make_ibs_register_decorator

from wbia.algo.detect.assigner import gid_keyed_assigner_results, gid_keyed_ground_truth, illustrate_all_assignments

import utool as ut
import numpy as np
import random
import os
from collections import OrderedDict, defaultdict
from datetime import datetime
import time

# illustration imports
from shutil import copy
from PIL import Image, ImageDraw
import wbia.plottool as pt


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
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


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


CLASSIFIER_OPTIONS = [
    {
        "name": "Nearest Neighbors",
        "clf": KNeighborsClassifier(3),
        "param_options": {
            'n_neighbors': [3, 5, 11, 19],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
        }
    },
    {
        "name": "Linear SVM",
        "clf": SVC(kernel="linear", C=0.025),
        "param_options": {
            'C': [1, 10, 100, 1000],
            'kernel': ['linear'],
        }
    },
    {
        "name": "RBF SVM",
        "clf": SVC(gamma=2, C=1),
        "param_options": {
            'C': [1, 10, 100, 1000],
            'gamma': [0.001, 0.0001],
            'kernel': ['rbf']
        },
    },
    {
        "name": "Decision Tree",
        "clf": DecisionTreeClassifier(),  # max_depth=5
        "param_options": {
            'max_depth': np.arange(1, 12),
            'max_leaf_nodes': [2, 5, 10, 20, 50, 100]
        }
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
        "name": "AdaBoost",
        "clf": AdaBoostClassifier(),
        "param_options": {
            'n_estimators': np.arange(10, 310, 50),
            'learning_rate': [0.01, 0.05, 0.1, 1],
        }
    },
    {
        "name": "Naive Bayes",
        "clf": GaussianNB(),
        "param_options": {}  # no hyperparams to optimize
    },
    {
        "name": "QDA",
        "clf": QuadraticDiscriminantAnalysis(),
        "param_options": {
            'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
    }
]


classifier_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                    "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                    "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


slow_classifier_names = "Gaussian Process"
slow_classifiers = GaussianProcessClassifier(1.0 * RBF(1.0)),


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
def compare_ass_classifiers(ibs, depc_table_name='assigner_viewpoint_features', print_accs=False):

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
        print("Tuning %s" % classifier['name'])
        accuracy, best_params = ibs._tune_grid_search(classifier['clf'], classifier['param_options'], assigner_data)
        print()
        accuracies[classifier['name']] = {
            'accuracy': accuracy,
            'best_params': best_params
        }
        if accuracy > best_acc:
            best_acc = accuracy
            best_clf_name = classifier['name']
            best_clf_params = best_params

    print('best performance: %s using %s with params %s' %
          (best_acc, best_clf_name, best_clf_params))

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
    print("Tune Fit Time: %s" % (end - start))
    pred = tune_search.predict(X_test)
    accuracy = np.count_nonzero(np.array(pred) == np.array(y_test)) / len(pred)
    print("Tune Accuracy: %s" % accuracy)
    print("best parms   : %s" % tune_search.best_params_)

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
    print("Tune Fit Time: %s" % (end - start))
    pred = tune_search.predict(X_test)
    accuracy = np.count_nonzero(np.array(pred) == np.array(y_test)) / len(pred)
    print("Tune Accuracy: %s" % accuracy)
    print("best parms   : %s" % tune_search.best_params_)

    return accuracy, tune_search.best_params_


# for wild dog dev
@register_ibs_method
def wd_assigner_data(ibs):
    return wd_training_data('part_assignment_features')


@register_ibs_method
def wd_normed_assigner_data(ibs):
    return wd_training_data('normalized_assignment_features')


@register_ibs_method
def wd_training_data(ibs, depc_table_name='assigner_viewpoint_features', balance_t_f=True):
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
    pairs_in_train = ibs.gid_train_test_split(all_pairs[0])  # we could pass just the pair aids or just the body aids bc gids are the same
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


    assigner_data = {'data': train_feats, 'target': train_truth,
                     'test': test_feats, 'test_truth': test_truth,
                     'train_pairs': train_pairs, 'test_pairs': test_pairs}

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
    keep_flags = [gt or (i in subsampled_false_indices) for i, gt in enumerate(ground_truth)]
    return keep_flags


def train_test_split(item_list, random_seed=777, test_size=0.1):
    import random
    import math
    random.seed(random_seed)
    sample_size = math.floor(len(item_list) * test_size)
    all_indices = list(range(len(item_list)))
    test_indices = random.sample(all_indices, sample_size)
    test_items = [item_list[i] for i in test_indices]
    train_indices = sorted(list(
        set(all_indices) - set(test_indices)
    ))
    train_items = [item_list[i] for i in train_indices]
    return train_items, test_items


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

    gid_to_assigner_results = gid_keyed_assigner_results(ibs, all_pairs, all_unassigned_aids)
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
                false_neg_log_index = min(false_negatives - 2, max_allowed_errors - 1)  # ie, if we have 2 errors, we have a false neg even allowing 1 error, in index 0 of that list
                try:
                    gids_with_false_neg_allowing_errors[false_neg_log_index] += 1
                except:
                    ut.embed()

        n_false_positives += false_positives
        if false_positives > 0:
            gids_with_false_positives += 1
        if false_negatives > 0 and false_positives > 0:
            gids_with_both_errors += 1

        pairs_equal = sorted(gid_to_assigner_results[gid]['pairs']) == sorted(gid_to_ground_truth[gid]['pairs'])
        if pairs_equal:
            correct_gids += [gid]
        else:
            incorrect_gids += [gid]

    n_gids = len(gid_to_assigner_results.keys())
    accuracy = len(correct_gids) / n_gids
    incorrect_gids = n_gids - len(correct_gids)
    acc_allowing_errors = [1 - (nerrors / n_gids)
                           for nerrors in gids_with_false_neg_allowing_errors]
    print('accuracy with cutoff of %s: %s' % (cutoff_score, accuracy))
    for i, acc_allowing_error in enumerate(acc_allowing_errors):
        print('        allowing %s errors, acc = %s' % (i + 1, acc_allowing_error))
    print('        %s false positives on %s error images' % (n_false_positives, gids_with_false_positives))
    print('        %s false negatives on %s error images' % (n_false_negatives, gids_with_false_negatives))
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

