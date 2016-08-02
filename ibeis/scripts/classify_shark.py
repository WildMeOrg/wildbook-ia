# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.svm
import sklearn.metrics
import sklearn.model_selection
from sklearn import preprocessing
import plottool as pt
from plottool import abstract_interaction
(print, rrr, profile) = ut.inject2(__name__, '[classify_shark]')


@ut.reloadable_class
class InteractAnnotClasses(abstract_interaction.AbstractInteraction):
    def __init__(inter, ibs, config, df_chunks, nCols, **kwargs):
        inter.df_chunks = df_chunks
        inter.ibs = ibs
        inter.nCols = nCols
        inter.config = config
        inter._init_lists()
        super(InteractAnnotClasses, inter).__init__(**kwargs)

    def _init_lists(inter):
        inter.offset_lists = []
        inter.sizes_lists = []
        inter.sf_lists = []
        inter.ax_list = []
        inter.metadata_lists = []
        inter.data_lists = []

    def _show_page(inter):
        from ibeis_cnn import draw_results
        inter._ensure_running()
        inter._init_lists()

        pnum_ = pt.make_pnum_nextgen(nCols=inter.nCols, nSubplots=len(inter.df_chunks))
        for df_chunk in inter.df_chunks:
            ibs = inter.ibs
            config = inter.config
            annots_chunk = ibs.annots(df_chunk['aid'].values, config=config)
            #data_lists = [(np.array(annots_chunk.hog_img) * 255).astype(np.uint8),
            #              annots_chunk.chips]
            #data_lists = [annots_chunk.chips]
            data_lists = [ibs.get_image_thumbnail(annots_chunk.gids, draw_annots=True)]
            label_list = (1 - df_chunk['failed']).values
            flat_metadata = df_chunk.to_dict(orient='list')
            flat_metadata['tags'] = annots_chunk.case_tags
            tup = draw_results.get_patch_chunk(data_lists, label_list,
                                               flat_metadata,
                                               draw_meta=['decision', 'tags'],
                                               vert=False, fontScale=4.0)
            inter.metadata_lists.append(flat_metadata)
            img, offset_list, sf_list, sizes_list = tup
            inter.offset_lists.append(offset_list)
            inter.sf_lists.append(sf_list)
            inter.sizes_lists.append(sizes_list)
            inter.data_lists.append(data_lists)
            fig, ax = pt.imshow(img, fnum=inter.fnum, pnum=pnum_())
            ax.set_title(df_chunk.nice)
            inter.ax_list.append(ax)
        pt.adjust_subplots2(top=.95, left=0, right=1, bottom=.00, hspace=.1, wspace=0)

    def on_click_inside(inter, event, ax):
        print('click inside')

        def get_label_index(inter, event, ax):
            """ generalize """
            x, y = event.xdata, event.ydata
            def find_offset_index(offsets, sizes, x, y):
                x1_pts = offsets.T[0]
                x2_pts = offsets.T[0] + sizes.T[0]
                y1_pts = offsets.T[1]
                y2_pts = offsets.T[1] + sizes.T[1]

                in_bounds = np.logical_and.reduce([
                    x >= x1_pts, x < x2_pts,
                    y >= y1_pts, y < y2_pts
                ])
                valid_idxs = np.where(in_bounds)[0]
                #print('valid_idxs = %r' % (valid_idxs,))
                assert len(valid_idxs) == 1
                return valid_idxs[0]
            try:
                # Find plot axes index
                px = inter.ax_list.index(ax)
            except ValueError:
                label_index = None
            else:
                offsets = np.array(inter.offset_lists[px])
                sfs = np.array(inter.sf_lists[px])
                orig_sizes = np.array(inter.sizes_lists[px])
                sizes = np.array(orig_sizes) * sfs
                data_list = inter.data_lists[px]
                data_per_label = len(data_list)
                #num_rows = data_per_label
                num_cols = (len(offsets) // data_per_label)
                _subindex = find_offset_index(offsets, sizes, x, y)
                row_index = _subindex // num_cols
                col_index = _subindex % num_cols
                #label_index = inter.multiindicies[plot_index][row_index]
                print('_subindex = %r' % (_subindex,))
                print('row_index = %r' % (row_index,))
                print('col_index = %r' % (col_index,))
                print('px = %r' % (px,))
                #print('label_index = %r' % (label_index,))
                label_index = (px, col_index)
            return label_index

        options = []

        label_index = get_label_index(inter, event, ax)

        if label_index is not None:
            px, col_index = label_index
            flat_metadata = inter.metadata_lists[px]

            if 'aid' in flat_metadata:
                from ibeis.viz.interact import interact_chip
                ibs = inter.ibs
                aid = flat_metadata['aid'][col_index]
                if ibs is not None:
                    options += interact_chip.build_annot_context_options(ibs, aid)
                    print('Annot Info:' + ut.repr3(inter.ibs.get_annot_info([aid], case_tags=True)))

        if event.button == 3:
            #options = inter.context_option_funcs[index]()
            options += [
                ('Present', pt.present),
            ]
            inter.show_popup_menu(options, event)


def get_sharks_dataset(target_type=None):
    """
        >>> from ibeis.scripts.classify_shark import *  # NOQA
    """
    import ibeis
    ibs = ibeis.opendb('WS_ALL')
    config = {
        'dim_size': (256, 256),
        'resize_dim': 'wh'
    }
    all_annots = ibs.annots(config=config)

    TARGET_TYPE = 'binary'
    #TARGET_TYPE = 'multiclass1'
    if target_type is None:
        target_type = TARGET_TYPE

    orig_case_tags = all_annots.case_tags
    tag_vocab = ut.flat_unique(*orig_case_tags)
    print('Original tags')
    print(ut.repr3(ut.dict_hist(ut.flatten(orig_case_tags))))

    def cleanup_tags(orig_case_tags, tag_vocab):
        regex_map = [
            ('injur-trunc', 'injur-trunc'),
            ('trunc', 'injur-trunc'),
            ('healthy', 'healthy'),
            (['injur-unknown', 'other_injury'], 'injur-other'),
            ('nicks', 'injur-nicks'),
            ('scar', 'injur-scar'),
            ('bite', 'injur-bite'),
            ('pose:novel', None),
        ]
        alias_map = ut.build_alias_map(regex_map, tag_vocab)
        unmapped = list(set(tag_vocab) - set(alias_map.keys()))
        case_tags = ut.alias_tags(orig_case_tags, alias_map)
        print('unmapped = %r' % (unmapped,))
        return case_tags

    case_tags = cleanup_tags(orig_case_tags, tag_vocab)

    print('Cleaned tags')
    print(ut.repr3(ut.dict_hist(ut.flatten(case_tags))))

    ntags_list = np.array(ut.lmap(len, case_tags))
    is_no_tag = ntags_list == 0
    is_single_tag = ntags_list == 1
    is_multi_tag = ntags_list > 1

    if target_type == 'binary':
        regex_map = [
            ('injur-.*', 'injured'),
            ('healthy', 'healthy'),
        ]
        tag_vocab = ut.flat_unique(*case_tags)
        alias_map = ut.build_alias_map(regex_map, tag_vocab)
        case_tags2 = ut.alias_tags(case_tags, alias_map)
    elif target_type == 'multiclass1':
        regex_map = [
            ('injur-trunc', 'injur-trunc'),
            ('healthy', 'healthy'),
            ('injur-.*', 'injur-other'),
        ]
        tag_vocab = ut.flat_unique(*case_tags)
        alias_map = ut.build_alias_map(regex_map, tag_vocab)
        unmapped = list(set(tag_vocab) - set(alias_map.keys()))
        print('unmapped = %r' % (unmapped,))

        case_tags2 = ut.alias_tags(case_tags, alias_map)
    elif target_type == 'multiclass2':
        regex_map = [
            ('injur-trunc', 'injur-trunc'),
            ('healthy', 'healthy'),
            ('injur-.*', None),
        ]
        tag_vocab = ut.flat_unique(*case_tags)
        alias_map = ut.build_alias_map(regex_map, tag_vocab)
        unmapped = list(set(tag_vocab) - set(alias_map.keys()))
        print('unmapped = %r' % (unmapped,))

        case_tags2 = ut.alias_tags(case_tags, alias_map)
    elif target_type == '_experimental_multilabel':
        pass
        # Binarize into multi-class labels
        # http://stackoverflow.com/questions/10526579/use-scikit-learn-to-classify-into-multiple-categories
        #menc = preprocessing.MultiLabelBinarizer()
        #menc.fit(annot_tags)
        #target = menc.transform(annot_tags)
        #enc = menc
    else:
        raise ValueError('Unknown target_type=%r' % (target_type,))
        # henc = preprocessing.OneHotEncoder()
        # henc.fit(menc.transform(annot_tags))
        # target = henc.transform(menc.transform(annot_tags))
        # target = np.array([int('healthy' not in tags) for tags in annots.case_tags])

    ntags_list = np.array(ut.lmap(len, case_tags2))
    is_no_tag = ntags_list == 0
    is_single_tag = ntags_list == 1
    is_multi_tag = ntags_list > 1

    print('Multi Tags: %s' % (ut.repr2(ut.compress(case_tags2, is_multi_tag), nl=1),))
    multi_annots = all_annots.compress(is_multi_tag)  # NOQA
    #ibs.set_image_imagesettext(multi_annots.gids, ['MultiTaged'] * is_multi_tag.sum())

    print('can\'t use %r annots due to no labels' % (is_no_tag.sum(),))
    print('can\'t use %r annots due to inconsistent labels' % (is_multi_tag.sum(),))
    print('will use %r annots with consistent labels' % (is_single_tag.sum(),))

    annot_tags = ut.compress(case_tags2, is_single_tag)
    annots = all_annots.compress(is_single_tag)
    annot_tag_hist = ut.dict_hist(ut.flatten(annot_tags))
    print('Final Annot Tags')
    print(ut.repr3(annot_tag_hist))

    # target_names = ['healthy', 'injured']
    enc = preprocessing.LabelEncoder()
    enc.fit(ut.unique(ut.flatten(annot_tags)))
    target = enc.transform(ut.flatten(annot_tags))
    target_names = enc.classes_

    data = np.array([h.ravel() for h in annots.hog_hog])

    # Build scipy / scikit data standards
    ds = sklearn.datasets.base.Bunch(
        ibs=ibs,
        aids=annots.aids,
        name='sharks',
        DESCR='injured-vs-healthy whale sharks',
        data=data,
        target=target,
        target_names=target_names,
        target_labels=enc.transform(target_names),
        enc=enc,
        config=config
    )
    return ds


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
class ClfProblem(object):
    """
    Harness for researching a classification problem
    """
    def __init__(problem, ds):
        problem.ds = ds

    def print_support_info(problem):
        enc = problem.ds.enc
        target_labels = enc.inverse_transform(problem.ds.target)
        label_hist = ut.dict_hist(target_labels)
        print('support hist' + ut.repr3(label_hist))

    def fit_new_classifier(problem, train_idx):
        """
        http://leon.bottou.org/research/stochastic
        http://blog.explainmydata.com/2012/06/ntrain-24853-ntest-25147-ncorrupt.html
        http://scikit-learn.org/stable/modules/svm.html#svm-classification
        """
        print('[problem] train classifier on %d data points' % (len(train_idx)))
        data = problem.ds.data
        target = problem.ds.target
        x_train = data.take(train_idx, axis=0)
        y_train = target.take(train_idx, axis=0)
        clf = sklearn.svm.SVC(kernel='linear', C=1, class_weight='balanced',
                              decision_function_shape='ovr')
        # C, penalty, loss

        #param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        #              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        #param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        #              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        #clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
        #clf = clf.fit(X_train_pca, y_train)

        clf.fit(x_train, y_train)
        return clf

    def test_classifier(problem, clf, test_idx):
        print('[problem] test classifier on %d data points' % (len(test_idx),))
        data = problem.ds.data
        target = problem.ds.target
        x_test = data.take(test_idx, axis=0)
        y_true = target.take(test_idx, axis=0)

        if len(clf.classes_) == 2:
            # Adapt _ovr_decision_function for 2-class case
            # This is simply a linear scaling into a probability based on the
            # other members of this query.
            X = clf._validate_for_predict(x_test)
            X = clf._compute_kernel(X)
            _dec2 = clf._dense_decision_function(X)
            dec2 = -_dec2

            n_samples = dec2.shape[0]
            n_classes = len(clf.classes_)
            final = np.zeros((n_samples, n_classes))
            confidence_max = max(np.abs(dec2.max()), np.abs(dec2.min()))
            norm_conf = ((dec2.T[0] / confidence_max) + 1) / 2
            final.T[0] = 1 - norm_conf
            final.T[1] = norm_conf
            # output comparable to multiclass version
            y_conf = final
        else:
            # Get notion of confidence / probability of decision
            y_conf = clf.decision_function(x_test)

        y_pred = y_conf.argmax(axis=1)
        #if False:
        #    real_pred = clf.predict(x_test)
        #    real_conf = clf.decision_function(x_test)
        #    np.all(y_pred == real_pred)
        #    np.all((real_conf > 0) == real_pred)
        #    np.all((norm_conf > 0) == real_pred)
        #    assert np.all(dec2.ravel() == real_conf)

        result = ClfSingleResult(problem.ds, test_idx, y_true, y_pred, y_conf)
        return result

    def stratified_2sample_idxs(problem, frac=.2, split_frac=.75):
        target = problem.ds.target
        target_labels = problem.ds.target_labels

        rng = np.random.RandomState(043)
        train_sample = []
        test_sample = []
        for label in target_labels:
            target_idxs = np.where(target == label)[0]
            subset_size = int(len(target_idxs) * frac)
            rand_idx = ut.random_indexes(len(target_idxs), subset_size, rng=rng)
            sample_idx = ut.take(target_idxs, rand_idx)
            split = int(len(sample_idx) * split_frac)
            train_sample.append(sample_idx[split:])
            test_sample.append(sample_idx[:split])

        train_idx = np.array(sorted(ut.flatten(train_sample)))
        test_idx = np.array(sorted(ut.flatten(test_sample)))
        return train_idx, test_idx

    def gen_crossval_idxs(problem, n_folds=2):
        xvalkw = dict(n_folds=n_folds, shuffle=True, random_state=43432)
        target = problem.ds.target
        #import sklearn.cross_validation
        #skf = sklearn.cross_validation.StratifiedKFold(target, **xvalkw)
        #_iter = skf
        skf = sklearn.model_selection.StratifiedKFold(**xvalkw)
        _iter = skf.split(y=target)
        msg = 'cross-val test on %s' % (problem.ds.name)
        for count, (train_idx, test_idx) in enumerate(ut.ProgIter(_iter, lbl=msg)):
            yield train_idx, test_idx


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

        # passed = y_pred == y_true
        # failed = y_pred != y_true
        #confusion = sklearn.metrics.confusion_matrix(y_true, y_pred)

        # is_tn = np.logical_and(passed, y_true == 0)
        # is_fp = np.logical_and(failed, y_true == 0)
        # is_fn = np.logical_and(failed, y_true == 1)
        # is_tp = np.logical_and(passed, y_true == 1)

        # columns = ['tn', 'fp', 'fn', 'tp', 'decision', 'pred']
        # column_data = [is_tn, is_fp, is_fn, is_tp, y_conf, y_pred]

        index = pd.Series(test_idx, name='test_idx')
        if len(result.ds.target_names) == 1:
            y_conf
        decision = pd.DataFrame(y_conf, index=index, columns=result.ds.target_names)
        result.decision = decision / 3
        easiness = np.array(ut.ziptake(result.decision.values, y_true))
        columns = ['pred', 'easiness']
        column_data = [y_pred, easiness]
        data = dict(zip(columns, column_data))
        result.df = pd.DataFrame(data, index, columns)

        y_true
        #result.decision = pd.Series(y_conf, index, name='decision', dtype=np.float)

        #result._compiled['confusion'] = confusion
        #score = (1 - (sum(passed) / len(passed)))
        #result._compiled['score'] = score

    def print_report(result):
        report = sklearn.metrics.classification_report(
            result.y_true, result.y_pred,
            target_names=result.ds.target_names)
        print(report)


def stratified_sample_idxs_balanced(target, frac=.2, balanced=True):
    rng = np.random.RandomState(43)
    sample = []
    for label in np.unique(target):
        target_idxs = np.where(target == label)[0]
        subset_size = int(len(target_idxs) * frac)
        rand_idx = ut.random_indexes(len(target_idxs), subset_size, rng=rng)
        sample_idx = ut.take(target_idxs, rand_idx)
        sample.append(sample_idx)
    sample_idx = np.array(sorted(ut.flatten(sample)))
    return sample_idx


def stratified_sample_idxs_unbalanced(target, size=1000):
    rng = np.random.RandomState(43)
    sample = []
    for label in np.unique(target):
        target_idxs = np.where(target == label)[0]
        subset_size = size
        rand_idx = ut.random_indexes(len(target_idxs), subset_size, rng=rng)
        sample_idx = ut.take(target_idxs, rand_idx)
        sample.append(sample_idx)
    sample_idx = np.array(sorted(ut.flatten(sample)))
    return sample_idx


def learn_injured_sharks():
    r"""
    References:
        http://scikit-learn.org/stable/model_selection.html

    TODO:
        * Change unreviewed healthy tags to healthy-likely

    Example:
        >>> from ibeis.scripts.classify_shark import *  # NOQA
        >>> learn_injured_sharks()
    """
    from ibeis.scripts import classify_shark
    import sklearn.metrics
    import plottool as pt
    import pandas as pd

    pt.qt4ensure()

    target_type = 'binary'
    #target_type = 'multiclass1'
    #target_type = 'multiclass2'
    ds = classify_shark.get_sharks_dataset(target_type)

    # Sample the dataset
    #idxs = classify_shark.stratified_sample_idxs_balanced(ds.target, .5)
    #idxs = classify_shark.stratified_sample_idxs_unbalanced(ds.target, 1500)
    #idxs = classify_shark.stratified_sample_idxs_unbalanced(ds.target, 1000)
    #ds.target = ds.target.take(idxs, axis=0)
    #ds.data = ds.data.take(idxs, axis=0)
    #ds.aids = ut.take(ds.aids, idxs)

    problem = classify_shark.ClfProblem(ds)
    problem.print_support_info()

    result_list = []
    #train_idx, test_idx = problem.stratified_2sample_idxs()
    n_folds = 5
    for train_idx, test_idx in problem.gen_crossval_idxs(n_folds):
        clf = problem.fit_new_classifier(train_idx)
        result = problem.test_classifier(clf, test_idx)
        result_list.append(result)

    for result in result_list:
        result.compile_results()

    for result in result_list:
        result.print_report()

    isect_sets = [set(s1).intersection(set(s2)) for s1, s2 in ut.combinations([
        result.df.index for result in result_list], 2)]
    assert all([len(s) == 0 for s in isect_sets]), ('cv sets should not intersect')

    pd.set_option("display.max_rows", 20)

    # Combine information from results
    df = pd.concat([result.df for result in result_list])
    df['hardness'] = 1 / df['easiness']
    df['aid'] = ut.take(ds.aids, df.index)
    df['target'] = ut.take(ds.target, df.index)
    df['failed'] = df['pred'] != df['target']

    report = sklearn.metrics.classification_report(
        y_true=df['target'], y_pred=df['pred'],
        target_names=result.ds.target_names)
    print(report)

    confusion = sklearn.metrics.confusion_matrix(df['target'], df['pred'])
    print('Confusion Matrix:')
    print(pd.DataFrame(confusion, columns=[' ' + m for m in result.ds.target_names],
                       index=['gt ' + m for m in result.ds.target_names]))

    def snapped_slice(size, frac, n):
        start = int(size * frac - np.ceil(n / 2))
        stop  = int(size * frac + np.floor(n / 2))
        if stop >= size:
            buf = (stop - size + 1)
            start -= buf
            stop -= buf
        if start < 0:
            buf = 0 - start
            stop += buf
            start += buf
        assert stop < size, 'out of bounds'
        sl = slice(start, stop)
        return sl

    def grab_subchunk(frac, n, target):
        df_chunk = df.take(df['hardness'].argsort())
        if target is not None:
            df_chunk = df_chunk[df_chunk['target'] == target]
        #df_chunk = df_chunk[df_chunk[err] > 0]
        #frac = {'start': 0.0, 'middle': 0.5, 'end': 1.0}[place]
        sl = snapped_slice(len(df_chunk), frac, n)
        idx = df_chunk.index[sl]
        df_chunk = df_chunk.loc[idx]
        place_name = 'hardness=%.2f' % (frac,)
        if target is not None:
            df_chunk.nice = place_name + ' ' + ds.target_names[target]
        else:
            df_chunk.nice = place_name
        return df_chunk

    n = 5
    #places = ['start', 'middle', 'end']
    #fracs = [0.0, .5, 1.0]
    fracs = [.7, .8, .9, 1.0]
    df_chunks = [grab_subchunk(frac, n, target)
                 for frac in fracs for target in ds.target_labels]

    #from ibeis_cnn import draw_results
    ibs = ds.ibs
    config = ds.config

    inter = classify_shark.InteractAnnotClasses(ibs, config, df_chunks, nCols=len(ds.target_labels))
    inter.start()

    #fnum = 1
    #pnum_ = pt.make_pnum_nextgen(nRows=len(places), nSubplots=len(df_chunks))
    #for df_chunk in df_chunks:
    #    if len(df_chunk) == 0:
    #        import vtool as vt
    #        img = vt.get_no_symbol(size=(n * 100, 200))
    #        #size=(200, 100))
    #        #img = np.zeros((10, 10), dtype=np.uint8)
    #    else:
    #        annots_chunk = ibs.annots(df_chunk['aid'].values, config=config)
    #        data_lists = [(np.array(annots_chunk.hog_img) * 255).astype(np.uint8),
    #                      annots_chunk.chips]
    #        label_list = (1 - df_chunk['failed']).values
    #        flat_metadata = df_chunk.to_dict(orient='list')
    #        flat_metadata['tags'] = annots_chunk.case_tags
    #        tup = draw_results.get_patch_chunk(data_lists, label_list,
    #                                           flat_metadata,
    #                                           draw_meta=['decision', 'tags'],
    #                                           vert=False, fontScale=4.0)
    #        img, offset_list, sf_list, stacked_orig_sizes = tup
    #    fig, ax = pt.imshow(img, fnum=fnum, pnum=pnum_())
    #    ax.set_title(df_chunk.nice)
    #pt.adjust_subplots2(top=.95, left=0, right=1, bottom=.00, hspace=.1, wspace=0)

    #if False:
    #    pt.qt4ensure()
    #    subset_df = df_chunk
    #    for idx in ut.InteractiveIter(subset_df.index.values):
    #        dfrow = subset_df.loc[idx]
    #        assert dfrow['aid'] == ds.aids[idx]
    #        annot = ibs.annots([dfrow['aid']], config=config)
    #        hogimg = annot.hog_img[0]
    #        chip = annot.chips[0]
    #        pt.clf()
    #        pt.imshow(hogimg, pnum=(1, 2, 1))
    #        pt.imshow(chip, pnum=(1, 2, 2))
    #        pt.set_xlabel(str(annot.case_tags[0]))
    #        fig = pt.gcf()
    #        print(dfrow)
    #        fig.show()
    #        fig.canvas.draw()

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
