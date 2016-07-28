# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.svm
import sklearn.metrics
import sklearn.cross_validation
(print, rrr, profile) = ut.inject2(__name__, '[classify_shark]')


#@ut.reloadable_class
#class ClfMultiResult(object):
#    def __init__(multi_result, result_list):
#        multi_result.result_list = result_list

#    def compile_results(multi_result):
#        import pandas as pd
#        result_list = multi_result.result_list
#        multi_result.df = reduce(ut.partial(pd.DataFrame.add, fill_value=0), [result.df for result in result_list])
#        #hardness = 1 / multi_result.df['decision'].abs()

#    def get_hardest_fail_idxs(multi_result):
#        df = multi_result.df
#        sortx = multi_result.hardness.argsort()[::-1]
#        # Order by hardness
#        df = multi_result.df.take(sortx)
#        failed = multi_result.df['is_fp'] + multi_result.df['is_fn']
#        # Grab only failures
#        hard_fail_idxs = failed[failed > 0].index.values
#        return hard_fail_idxs


@ut.reloadable_class
class ClfSingleResult(object):
    r"""
    Reports the results of a classification problem

    Example:
        >>> result = ClfSingleResult()
    """
    def __init__(result, ds=None, test_idx=None, y_true=None, y_pred=None, y_conf=None):
        result.ds = ds
        result.test_idx = test_idx
        result.y_true = y_true
        result.y_pred = y_pred
        result.y_conf = y_conf

    def compile_results(result):
        import pandas as pd
        y_true = result.y_true
        y_pred = result.y_pred
        y_conf = result.y_conf
        test_idx = result.test_idx

        passed = y_pred == y_true
        failed = y_pred != y_true
        #confusion = sklearn.metrics.confusion_matrix(y_true, y_pred)

        is_tn = np.logical_and(passed, y_true == 0)
        is_fp = np.logical_and(failed, y_true == 0)
        is_fn = np.logical_and(failed, y_true == 1)
        is_tp = np.logical_and(passed, y_true == 1)

        columns = ['is_tn', 'is_fp', 'is_fn', 'is_tp', 'decision']
        column_data = [is_tn, is_fp, is_fn, is_tp, y_conf]
        data = dict(zip(columns, column_data))

        index = pd.Series(test_idx, name='test_idx')
        result.df = pd.DataFrame(data, index, columns)
        #result.decision = pd.Series(y_conf, index, name='decision', dtype=np.float)

        #result._compiled['confusion'] = confusion
        #score = (1 - (sum(passed) / len(passed)))
        #result._compiled['score'] = score

    def print_report(result):
        report = sklearn.metrics.classification_report(
            result.y_true, result.y_pred,
            target_names=result.ds.target_names)
        print(report)


@ut.reloadable_class
class ClfProblem(object):
    """
    Harness for researching a classification problem
    """
    def __init__(problem, ds):
        problem.ds = ds

    def print_support_info(problem):
        target_to_num = ut.dict_hist(problem.ds.target)
        hist = ut.map_dict_keys(ut.partial(ut.take, problem.ds.target_names), target_to_num)
        print('support hist' + ut.repr3(hist))

    def fit_new_classifier(problem, train_idx):
        print('[problem] train classifier')
        data = problem.ds.data
        target = problem.ds.target
        x_train = data.take(train_idx, axis=0)
        y_train = target.take(train_idx, axis=0)
        clf = sklearn.svm.SVC(kernel='linear', C=1)
        clf.fit(x_train, y_train)
        return clf

    def test_classifier(problem, clf, test_idx):
        print('[problem] test classifier')
        data = problem.ds.data
        target = problem.ds.target
        x_test = data.take(test_idx, axis=0)
        y_true = target.take(test_idx, axis=0)
        y_pred = clf.predict(x_test)
        # Get notion of confidence / probability of decision
        y_conf = clf.decision_function(x_test)
        result = ClfSingleResult(problem.ds, test_idx, y_true, y_pred, y_conf)
        return result

    def gen_crossval_idxs(problem):
        xvalkw = dict(n_folds=2, shuffle=True, random_state=43432)
        target = problem.ds.target
        kf = sklearn.cross_validation.StratifiedKFold(target, **xvalkw)
        msg = 'cross-val test on %s' % (problem.ds.name)
        for train_idx, test_idx in ut.ProgIter(kf, lbl=msg):
            yield train_idx, test_idx


def learn_injured_sharks():
    r"""
    References:
        http://scikit-learn.org/stable/modules/cross_validation.html

    Example:
        >>> from ibeis.scripts.classify_shark import *  # NOQA
    """
    import plottool as pt
    import pandas as pd
    import ibeis

    pt.qt4ensure()

    ibs = ibeis.opendb('WS_ALL')
    config = {
        'dim_size': (256, 256),
        'resize_dim': 'wh'
    }
    annots = ibs.annots(config=config)
    data = np.array([h.ravel() for h in annots.hog_hog])
    target = np.array([int('healthy' not in tags) for tags in annots.case_tags])
    # Build scipy / scikit data standards
    ds = sklearn.datasets.base.Bunch(
        ibs=ibs,
        aids=annots.aids,
        data=data,
        target=target,
        name='sharks',
        DESCR='injured-vs-healthy whale sharks',
        target_names=['healthy', 'injured'],
    )
    problem = ClfProblem(ds)
    problem.print_support_info()

    result_list = []
    for train_idx, test_idx in problem.gen_crossval_idxs():
        clf = problem.fit_new_classifier(train_idx)
        result = problem.test_classifier(clf, test_idx)
        result_list.append(result)

    for result in result_list:
        result.compile_results()

    for result in result_list:
        result.print_report()

    # Combine information from results
    #result_list = multi_result.result_list
    addfunc = ut.partial(pd.DataFrame.add, fill_value=0)
    df = reduce(addfunc, [result.df for result in result_list])
    df['easiness'] = df['decision'].abs()
    df['hardness'] = 1 / df['easiness']
    df['failed'] = df['is_fp'] + df['is_fn']
    df['aid'] = ut.take(ds.aids, df.index)
    df['target'] = ut.take(ds.target, df.index)

    report = sklearn.metrics.classification_report(
        y_true=df['target'], y_pred=df['decision'] > 0,
        target_names=result.ds.target_names)
    print(report)

    # Order by hardness and grab only failures
    hard_df = df.take(df['hardness'].argsort()[::-1])
    hard_df = hard_df[hard_df['failed'] > 0]
    # Order by easiness and grab only successes
    easy_df = df.take(df['easiness'].argsort()[::-1])
    easy_df = easy_df[easy_df['failed'] == 0]
    df1 = hard_df.take(range(6))
    df1.nice = 'Hard Failure Cases'
    df2 = easy_df.take(range(6))
    df2.nice = 'Easy Success Cases'

    def grab_subchunk(sortby, err, target, n):
        df_chunk = df.take(df[sortby].argsort()[::-1])
        if target is not None:
            df_chunk = df_chunk[df_chunk['target'] == target]
        df_chunk = df_chunk[df_chunk['is_' + err] > 0]
        if True:
            start = int(len(df_chunk) // 2 - np.ceil(n / 2))
            stop  = int(len(df_chunk) // 2 + np.floor(n / 2))
            sl = slice(start, stop)
            idx = df_chunk.index[sl]
            print('sl = %r' % (sl,))
            print('idx = %r' % (idx,))
            df_chunk = df_chunk.loc[idx]
            df_chunk.nice = err.upper()
        else:
            df_chunk = df_chunk.loc[df_chunk.index[slice(0, n)]]
            if target is not None:
                df_chunk.nice = {'easiness': 'Easy', 'hardness': 'Hard'}[sortby] + ' ' + err.upper() + ' ' + ds.target_names[target]
            else:
                df_chunk.nice = {'easiness': 'Easy', 'hardness': 'Hard'}[sortby] + ' ' + err.upper()
        return df_chunk

    #def confusion_by_label():
    # Print Confusion by label
    for target in [0, 1]:
        df_target = df[df['target'] == target]
        df_err = df_target[['is_tp', 'is_fp', 'is_fn', 'is_tn']]
        print('target = %r' % (ds.target_names[target]))
        print('df_err.sum() =%s' % (ut.repr3(df_err.sum().to_dict()),))

    #n = 3
    #df_list = [
    #    grab_subchunk('easiness', 'tn', 1, n),
    #    grab_subchunk('easiness', 'tn', 0, n),
    #    grab_subchunk('hardness', 'fp', 1, n),
    #    grab_subchunk('hardness', 'fp', 0, n),
    #    grab_subchunk('hardness', 'fn', 1, n),
    #    grab_subchunk('hardness', 'fn', 0, n),
    #    grab_subchunk('easiness', 'tp', 1, n),
    #    grab_subchunk('easiness', 'tp', 0, n),
    #]

    n = 4
    df_list = [
        grab_subchunk('easiness', 'tn', None, n),
        grab_subchunk('hardness', 'fp', None, n),
        grab_subchunk('hardness', 'fn', None, n),
        grab_subchunk('easiness', 'tp', None, n),
    ]

    from ibeis_cnn import draw_results
    fnum = 1
    pnum_ = pt.make_pnum_nextgen(nCols=len(df_list) // 2, nSubplots=len(df_list))
    for df_chunk in df_list:
        if len(df_chunk) == 0:
            import vtool as vt
            img = vt.get_no_symbol(size=(n * 100, 200))
            #size=(200, 100))
            #img = np.zeros((10, 10), dtype=np.uint8)
        else:
            annots_chunk = ibs.annots(df_chunk['aid'].values, config=config)
            data_lists = [(np.array(annots_chunk.hog_img) * 255).astype(np.uint8), annots_chunk.chips]
            label_list = (1 - df_chunk['failed']).values
            flat_metadata = df_chunk.to_dict(orient='list')
            flat_metadata['tags'] = annots_chunk.case_tags
            tup = draw_results.get_patch_chunk(data_lists, label_list, flat_metadata, draw_meta=['decision', 'tags'], vert=False, fontScale=4.0)
            img, offset_list, sf_list, stacked_orig_sizes = tup
        fig, ax = pt.imshow(img, fnum=fnum, pnum=pnum_())
        ax.set_title(df_chunk.nice)
    pt.adjust_subplots2(top=.95, left=0, right=1, bottom=.00, hspace=.1, wspace=0)

    if False:
        pt.qt4ensure()
        subset_df = df_chunk
        for idx in ut.InteractiveIter(subset_df.index.values):
            dfrow = subset_df.loc[idx]
            assert dfrow['aid'] == ds.aids[idx]
            annot = ibs.annots([dfrow['aid']], config=config)
            hogimg = annot.hog_img[0]
            chip = annot.chips[0]
            pt.clf()
            pt.imshow(hogimg, pnum=(1, 2, 1))
            pt.imshow(chip, pnum=(1, 2, 2))
            pt.set_xlabel(str(annot.case_tags[0]))
            fig = pt.gcf()
            print(dfrow)
            fig.show()
            fig.canvas.draw()

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.scripts.classify_shark
        python -m ibeis.scripts.classify_shark --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
