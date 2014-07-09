"""
This file is in a state of disarray. It is not included anywhere. Yet it might
have useful code in it
"""
from __future__ import absolute_import, division, print_function
import utool
(print, print_,  rrr, profile,
 printDBG) = utool.inject(__name__, '[classifier]', DEBUG=False)
# Science
import numpy as np
# HotSpotter
from ibeis.model.hots import report_results2 as rr2


def get_gt_cases(ibs):
    valid_aids = ibs.get_valid_aids()
    aid_list = [aid for aid in valid_aids if len(ibs.get_annot_groundtruth(aid)) > 0]
    return aid_list


def get_lblannoted_descriptors(allres, orgtype_='false'):
    qcxs = allres[orgtype_].qcxs
    aids  = allres[orgtype_].aids
    match_list = zip(qcxs, aids)
    aggdesc1, aggdesc2 = rr2.get_matching_descriptors(allres, match_list)
    return aggdesc1, aggdesc2


def train_classifier(ibs):
    aid_list = get_gt_cases(ibs)
    allres = rr2.get_allres(ibs, aid_list)
    neg_eg = get_lblannoted_descriptors(allres, 'false')
    pos_eg = get_lblannoted_descriptors(allres, 'true')

    # Cast to a datatype we can manipulate
    neg_eg = np.array(neg_eg, np.int32)
    pos_eg = np.array(pos_eg, np.int32)

    method = 3

    if method == 1:
        # Concatentation Method
        neg_vec = np.hstack([neg_eg[0], neg_eg[1]])
        pos_vec = np.hstack([pos_eg[0], pos_eg[1]])
    elif method == 2:
        # Multiplication Method (Cao and Snavely)
        neg_vec = neg_eg[0] * neg_eg[1]
        pos_vec = pos_eg[0] * pos_eg[1]
    elif method == 3:
        # My idea
        neg_vec = np.hstack([neg_eg[0] * neg_eg[1], (neg_eg[0] - neg_eg[1]) ** 2])
        pos_vec = np.hstack([pos_eg[0] * pos_eg[1], (pos_eg[0] - pos_eg[1]) ** 2])

    pos_train = pos_vec[0:len(pos_vec) // 2]
    neg_train = neg_vec[0:len(neg_vec) // 2]
    pos_lbl = np.ones((len(pos_train), 1), dtype=np.int32)
    neg_lbl = np.zeros((len(neg_train), 1), dtype=np.int32)
    pos_test = pos_vec[len(pos_vec) // 2:]  # NOQA
    neg_test = neg_vec[len(neg_vec) // 2:]  # NOQA
    train_data = np.vstack((pos_train, neg_train))  # NOQA
    train_lbl = np.vstack((pos_lbl, neg_lbl))  # NOQA


def test_classifier(classifier, pos_test, neg_test):
    pos_output = classifier.predict(pos_test)
    neg_output = classifier.predict(neg_test)

    tp_rate = pos_output.sum() / len(pos_output)
    fp_rate = neg_output.sum() / len(neg_output)
    print('tp_rate = %r' % tp_rate)
    print('fp_rate = %r' % fp_rate)


def train_random_forest(ibs, train_data, train_lbl, pos_test, neg_test):
    from sklearn.ensemble import RandomForestClassifier
    forest_parms = {
        'n_estimators': 10,
        'criterion': 'gini',
        'max_features': None,  # 'auto'
        'max_depth': None,
        'n_jobs': -1,
        'random_state': None,
        'verbose': True,
    }
    classifier = RandomForestClassifier(**forest_parms)
    classifier = classifier.fit(train_data, train_lbl)
    test_classifier(classifier, pos_test, neg_test)


def train_svm(ibs, train_data, train_lbl, pos_test, neg_test):
    from sklearn import svm
    classifier = svm.SVC(C=1.0,
                         cache_size=200,
                         class_weight=None,
                         coef0=0.0,
                         degree=3,
                         gamma=0.0,
                         kernel='rbf',
                         max_iter=-1,
                         probability=False,
                         shrinking=True,
                         tol=0.001,
                         verbose=True)
    classifier.fit(train_data, train_lbl)
    test_classifier(classifier, pos_test, neg_test)
