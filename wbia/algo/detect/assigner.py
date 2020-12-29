import logging
from os.path import expanduser, join
from wbia import constants as const
from wbia.control.controller_inject import register_preprocs, register_subprops, make_ibs_register_decorator
import utool as ut
import numpy as np
import random
import os
from collections import OrderedDict, defaultdict
from datetime import datetime
import time

from sklearn import preprocessing
from tune_sklearn import TuneGridSearchCV

# shitload of scikit classifiers
import numpy as np
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


# bunch of classifier models for training

(print, rrr, profile) = ut.inject2(__name__, '[assigner]')
logger = logging.getLogger('wbia')

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)

PARALLEL = not const.CONTAINERIZED
INPUT_SIZE = 224

ARCHIVE_URL_DICT = {}

INMEM_ASSIGNER_MODELS = {}

SPECIES_TO_ASSIGNER_MODELFILE = {
    'wild_dog': '/tmp/assigner_model.joblib',
    'wild_dog_dark': '/tmp/assigner_model.joblib',
    'wild_dog_light': '/tmp/assigner_model.joblib',
    'wild_dog_puppy': '/tmp/assigner_model.joblib',
    'wild_dog_standard': '/tmp/assigner_model.joblib',
    'wild_dog_tan': '/tmp/assigner_model.joblib',
}

CLASSIFIER_OPTIONS = [
    # {
    #     "name": "Nearest Neighbors",
    #     "clf": KNeighborsClassifier(3),
    #     "param_options": {
    #         'n_neighbors': [3,5,11,19],
    #         'weights': ['uniform', 'distance'],
    #         'metric': ['euclidean', 'manhattan'],
    #     }
    # },
    # {
    #     "name": "Linear SVM",
    #     "clf": SVC(kernel="linear", C=0.025),
    #     "param_options": {
    #         'C': [1, 10, 100, 1000],
    #         'kernel': ['linear'],
    #     }
    # },
    # {
    #     "name": "RBF SVM",
    #     "clf": SVC(gamma=2, C=1),
    #     "param_options": {
    #         'C': [1, 10, 100, 1000],
    #         'gamma': [0.001, 0.0001],
    #         'kernel': ['rbf']
    #     },
    # },
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
    #         'n_estimators': [200, 1000, 1500, 2000]
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
    # {
    #     "name": "AdaBoost",
    #     "clf": AdaBoostClassifier(),
    #     "param_options": {
    #          'n_estimators': np.arange(10, 310, 50),
    #          'learning_rate': [0.01, 0.05, 0.1, 1],
    #      }
    # },
    # {
    #     "name": "Naive Bayes",
    #     "clf": GaussianNB(),
    #     "param_options": {} # no hyperparams to optimize
    # },
    # {
    #     "name": "QDA",
    #     "clf": QuadraticDiscriminantAnalysis(),
    #     "param_options": {
    #         'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]
    #     }
    # }
]

# for model exploration
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
def compare_ass_classifiers(ibs, depc_table_name='theta_assignment_features', print_accs=False):

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
def tune_ass_classifiers(ibs, depc_table_name='theta_assignment_features'):

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
          best_acc, best_clf_name, best_clf_params)

    return accuracies


@register_ibs_method
def _tune_grid_search(ibs, clf, parameters, assigner_data=None):
    if assigner_data is None:
        assigner_data = ibs.wd_training_data()

    X_train = assigner_data['data']
    y_train = assigner_data['target']
    X_test  = assigner_data['test']
    y_test  = assigner_data['test_truth']

    tune_search = TuneGridSearchCV(
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
    X_test  = assigner_data['test']
    y_test  = assigner_data['test_truth']

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
def wd_training_data(ibs, depc_table_name='theta_assignment_features'):
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

    assigner_data = {'data': train_feats, 'target': train_truth,
                     'test': test_feats, 'test_truth': test_truth,
                     'train_pairs': train_pairs, 'test_pairs': test_pairs}

    return assigner_data


@register_ibs_method
def _are_part_annots(ibs, aid_list):
    species = ibs.get_annot_species(aid_list)
    are_parts = ['+' in specie for specie in species]
    return are_parts


def all_part_pairs(ibs, gid_list):
    all_aids = ibs.get_image_aids(gid_list)
    all_aids_are_parts = [ibs._are_part_annots(aids) for aids in all_aids]
    all_part_aids = [[aid for (aid, part) in zip(aids, are_parts) if part] for (aids, are_parts) in zip(all_aids, all_aids_are_parts)]
    all_body_aids = [[aid for (aid, part) in zip(aids, are_parts) if not part] for (aids, are_parts) in zip(all_aids, all_aids_are_parts)]
    part_body_parallel_lists = [_all_pairs_parallel(parts, bodies) for parts, bodies in zip(all_part_aids, all_body_aids)]
    all_parts  = [aid for part_body_parallel_list in part_body_parallel_lists
                  for aid in part_body_parallel_list[0]]
    all_bodies = [aid for part_body_parallel_list in part_body_parallel_lists
                  for aid in part_body_parallel_list[1]]
    return all_parts, all_bodies


def _all_pairs_parallel(list_a, list_b):
    pairs = [(a, b) for a in list_a for b in list_b]
    pairs_a = [pair[0] for pair in pairs]
    pairs_b = [pair[1] for pair in pairs]
    return pairs_a, pairs_b


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
        python -m algo.detect.assigner gid_train_test_split

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> from wbia.algo.detect.assigner import *
        >>> ibs = assigner_testdb_ibs()
        >>> #TODO: get input aids somehow
        >>> train_flag_list = ibs.gid_train_test_split(aids)
        >>> train_aids, test_aids = split_list(aids, train_flag_list)
        >>> train_gids = ibs.get_annot_gids(train_aids)
        >>> test_gids = ibs.get_annot_gids(test_aids)
        >>> train_gid_set = set(train_gids)
        >>> test_gid_set = set(test_gids)
        >>> assert len(train_gid_set & test_gid_set) is 0
        >>> assert len(train_gids) + len(test_gids) == len(aids)
    """
    print('calling gid_train_test_split')
    gid_list = ibs.get_annot_gids(aid_list)
    gid_set = list(set(gid_list))
    import random
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


@register_ibs_method
def assign_parts(ibs, all_aids, cutoff_score=0.5):
    gids = ibs.get_annot_gids(all_aids)
    gid_to_aids = defaultdict(list)
    for gid, aid in zip(gids, all_aids):
        gid_to_aids[gid] += [aid]

    all_pairs = []
    all_unassigned_aids = []

    for gid in gid_to_aids.keys():
        this_pairs, this_unassigned = _assign_parts_one_image(ibs, gid_to_aids[gid], cutoff_score)
        all_pairs += (this_pairs)
        all_unassigned_aids += this_unassigned

    return all_pairs, all_unassigned_aids



@register_ibs_method
def _assign_parts_one_image(ibs, aid_list, cutoff_score=0.5):

    are_part_aids = _are_part_annots(ibs, aid_list)
    part_aids = ut.compress(aid_list, are_part_aids)
    body_aids = ut.compress(aid_list, [not p for p in are_part_aids])

    gids = ibs.get_annot_gids(list(set(part_aids)) + list(set(body_aids)))
    num_images = len(set(gids))
    assert num_images is 1

    # parallel lists representing all possible part/body pairs
    all_pairs_parallel = _all_pairs_parallel(part_aids, body_aids)
    pair_parts, pair_bodies = all_pairs_parallel


    assigner_features   = ibs.depc_annot.get('theta_assignment_features', all_pairs_parallel)
    assigner_classifier = load_assigner_classifier(ibs, part_aids)

    assigner_scores = assigner_classifier.predict_proba(assigner_features)
    #  assigner_scores is a list of [P_false, P_true] probabilities which sum to 1, so here we just pare down to the true probabilities
    assigner_scores = [score[1] for score in assigner_scores]
    good_pairs, unassigned_aids = _make_assignments(pair_parts, pair_bodies, assigner_scores, cutoff_score)
    return good_pairs, unassigned_aids


def _make_assignments(pair_parts, pair_bodies, assigner_scores, cutoff_score=0.5):

    sorted_scored_pairs = [(part, body, score) for part, body, score in
                           sorted(zip(pair_parts, pair_bodies, assigner_scores),
                           key=lambda pbscore: pbscore[2], reverse=True)]

    assigned_pairs = []
    assigned_parts = set()
    assigned_bodies = set()
    n_bodies = len(set(pair_bodies))
    n_parts  = len(set(pair_parts))
    n_true_pairs = min(n_bodies, n_parts)
    for part_aid, body_aid, score in sorted_scored_pairs:
        assign_this_pair = part_aid not in assigned_parts and \
                           body_aid not in assigned_bodies and \
                           score >= cutoff_score

        if assign_this_pair:
            assigned_pairs.append((part_aid, body_aid))
            assigned_parts.add(part_aid)
            assigned_bodies.add(body_aid)

        if len(assigned_parts) is n_true_pairs \
           or len(assigned_bodies) is n_true_pairs \
           or score > cutoff_score:
            break

    unassigned_parts = set(pair_parts) - set(assigned_parts)
    unassigned_bodies = set(pair_bodies) - set(assigned_bodies)
    unassigned_aids = sorted(list(unassigned_parts) + list(unassigned_bodies))

    return assigned_pairs, unassigned_aids


def load_assigner_classifier(ibs, aid_list, fallback_species='wild_dog'):
    species_with_part = ibs.get_annot_species(aid_list[0])
    species = species_with_part.split('+')[0]
    if species in INMEM_ASSIGNER_MODELS.keys():
        clf = INMEM_ASSIGNER_MODELS[species]
    else:
        if species not in SPECIES_TO_ASSIGNER_MODELFILE.keys():
            print("WARNING: Assigner called for species %s which does not have an assigner modelfile specified. Falling back to the model for %s" % species, fallback_species)
            species = fallback_species

        model_fpath = SPECIES_TO_ASSIGNER_MODELFILE[species]
        from joblib import load
        clf = load(model_fpath)

    return clf


def check_accuracy(ibs, assigner_data, cutoff_score=0.5):

    all_aids = []
    for pair in assigner_data['test_pairs']:
        all_aids.extend(list(pair))
    all_aids = sorted(list(set(all_aids)))

    all_pairs, all_unassigned_aids = ibs.assign_parts(all_aids, cutoff_score)

    gid_to_assigner_results = gid_keyed_assigner_results(ibs, all_pairs, all_unassigned_aids)
    gid_to_ground_truth = gid_keyed_ground_truth(ibs, assigner_data)

    correct_gids = []
    incorrect_gids = []
    gids_with_false_positives = 0
    n_false_positives = 0
    gids_with_false_negatives = 0
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
        n_false_positives += false_positives
        if false_positives > 0:
            gids_with_false_positives += 1
        if false_negatives > 0 and false_positives > 0:
            gids_with_both_errors += 1

        pairs_equal = sorted(gid_to_assigner_results[gid]['pairs']) == sorted(gid_to_ground_truth[gid]['pairs'])
        if pairs_equal:
            correct_gids += [gid]
        else:
            incorrect_gids +=[gid]

    accuracy = len(correct_gids) / len(gid_to_assigner_results.keys())
    print('accuracy with cutoff of %s: %s' % (cutoff_score, accuracy))
    print('        %s false positives on %s error images' % (n_false_positives, gids_with_false_positives))
    print('        %s false negatives on %s error images' % (n_false_negatives, gids_with_false_negatives))
    print('        %s images with both errors' % (gids_with_both_errors))
    return accuracy


def gid_keyed_assigner_results(ibs, all_pairs, all_unassigned_aids):
    one_from_each_pair = [p[0] for p in all_pairs]
    pair_gids = ibs.get_annot_gids(one_from_each_pair)
    unassigned_gids = ibs.get_annot_gids(all_unassigned_aids)

    gid_to_pairs = defaultdict(list)
    for pair, gid in zip(all_pairs, pair_gids):
        gid_to_pairs[gid] += [pair]

    gid_to_unassigned = defaultdict(list)
    for aid, gid in zip(all_unassigned_aids, unassigned_gids):
        gid_to_unassigned[gid] += [aid]

    gid_to_assigner_results = {}
    for gid in (set(gid_to_pairs.keys()) | set(gid_to_unassigned.keys())):
        gid_to_assigner_results[gid] = {
            'pairs': gid_to_pairs[gid],
            'unassigned': gid_to_unassigned[gid]
        }

    return gid_to_assigner_results


def gid_keyed_ground_truth(ibs, assigner_data):
    test_pairs = assigner_data['test_pairs']
    test_truth = assigner_data['test_truth']
    assert len(test_pairs) == len(test_truth)

    aid_from_each_pair = [p[0] for p in test_pairs]
    gids_for_pairs = ibs.get_annot_gids(aid_from_each_pair)

    gid_to_pairs = defaultdict(list)
    gid_to_paired_aids = defaultdict(set)  # to know which have not been in any pair
    gid_to_all_aids = defaultdict(set)
    for pair, is_true_pair, gid in zip(test_pairs, test_truth, gids_for_pairs):
        gid_to_all_aids[gid] = gid_to_all_aids[gid] | set(pair)
        if is_true_pair:
            gid_to_pairs[gid] += [pair]
            gid_to_paired_aids[gid] = gid_to_paired_aids[gid] | set(pair)

    gid_to_unassigned_aids = defaultdict(list)
    for gid in gid_to_all_aids.keys():
        gid_to_unassigned_aids[gid] = list(gid_to_all_aids[gid] - gid_to_paired_aids[gid])


    gid_to_assigner_results = {}
    for gid in (set(gid_to_pairs.keys()) | set(gid_to_unassigned_aids.keys())):
        gid_to_assigner_results[gid] = {
            'pairs': gid_to_pairs[gid],
            'unassigned': gid_to_unassigned_aids[gid]
        }

    return gid_to_assigner_results












