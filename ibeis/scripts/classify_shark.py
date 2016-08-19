# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.svm
import sklearn.metrics
from sklearn import preprocessing
from ibeis_cnn.models import abstract_models
(print, rrr, profile) = ut.inject2(__name__, '[classify_shark]')


def shark_net(dry=False):
    """
    CommandLine:
        python -m ibeis.scripts.classify_shark shark_net
        python -m ibeis.scripts.classify_shark shark_net --dry
        python -m ibeis.scripts.classify_shark shark_net --vd

    Example:
        >>> from ibeis.scripts.classify_shark import *  # NOQA
        >>> shark_net()
    """
    from ibeis.scripts import classify_shark
    import ibeis
    ibs = ibeis.opendb('WS_ALL')  # NOQA
    config = {
        'dim_size': (224, 224),
        'resize_dim': 'wh'
    }

    # ------------
    # Define dataset
    # ------------
    target_type = 'binary'
    # ut.delete(ibs.get_neuralnet_dir())  # to reset
    dataset = classify_shark.get_shark_dataset(target_type, 'chip')

    # ------------
    # Define model
    # ------------
    model = classify_shark.WhaleSharkInjuryModel(
        name='injur-shark',
        dataset_dpath=dataset.dataset_dpath,
        training_dpath=ibs.get_neuralnet_dir(),
        #
        output_dims=2,
        data_shape=config['dim_size'] + (3,),
        batch_size=64,
    )
    model.initialize_architecture()
    model.print_layer_info()

    if False:
        state_fpath = model.get_model_state_fpath(dpath=model.trained_arch_dpath)
        model.load_model_state(fpath=state_fpath)
        X_test, y_test = dataset.subset('test')
        y_pred = model.predict(X_test)
        test_outptuts = model._predict(X_test)
        y_pred = test_outptuts['predictions']
        report = sklearn.metrics.classification_report(
            y_true=y_test, y_pred=y_pred,
        )
        print(report)

    if dry or ut.get_argflag('--dry'):
        return model, dataset

    model.learn_state.weight_decay = .01
    model.learn_state.learning_rate = .0001
    model.train_config.update(**dict(
        era_size=3,
        max_epochs=1200,
        rate_decay=.8,
        monitor=True,
    ))

    #model.build_backprop_func()
    #model.build_forward_func()

    # ---------------
    # Setup and learn
    # ---------------

    X_learn, y_learn = dataset.subset('learn')
    X_valid, y_valid = dataset.subset('valid')
    #model.ensure_training_state(X_learn, y_learn)
    model.fit(X_learn, y_learn, X_valid=X_valid, y_valid=y_valid)


@ut.reloadable_class
class WhaleSharkInjuryModel(abstract_models.AbstractCategoricalModel):
    """
    Example:
        >>> from ibeis.scripts.classify_shark import *  # NOQA
        >>> from ibeis.scripts import classify_shark
        >>> ds = classify_shark.get_sharks_dataset('binary', 'chip')
        >>> problem = classify_shark.ClfProblem(ds)
        >>> problem.print_support_info()
        >>> ibs = ds.ibs
    """

    def initialize_architecture(model, verbose=ut.VERBOSE, **kwargs):
        r"""

        CommandLine:
            python -m ibeis.scripts.classify_shark WhaleSharkInjuryModel.initialize_architecture
            python -m ibeis.scripts.classify_shark WhaleSharkInjuryModel.initialize_architecture --show

            python -m ibeis.scripts.classify_shark shark_net --dry --show
            python -m ibeis.scripts.classify_shark shark_net --vd

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.scripts.classify_shark import *  # NOQA
            >>> verbose = True
            >>> data_shape = tuple(ut.get_argval('--datashape', type_=list,
            >>>                                  default=(224, 224, 3)))
            >>> model = WhaleSharkInjuryModel(batch_size=64, output_dims=2,
            >>>                               data_shape=data_shape)
            >>> model.initialize_architecture()
            >>> model.print_model_info_str()
            >>> ut.quit_if_noshow()
            >>> model.show_arch()
            >>> ut.show_if_requested()
        """
        import ibeis_cnn.__LASAGNE__ as lasange
        from ibeis_cnn import custom_layers
        print('[model] initialize_architecture')
        lrelu = lasange.nonlinearities.LeakyRectify(leakiness=(1. / 10.))
        bundles = custom_layers.make_bundles(
            nonlinearity=lrelu, batch_norm=True,
            filter_size=(3, 3), stride=(1, 1),
            pool_size=(2, 2), pool_stride=(2, 2)
        )
        InputBundle          = bundles['InputBundle']
        ConvBundle           = bundles['ConvBundle']
        FullyConnectedBundle = bundles['FullyConnectedBundle']
        SoftmaxBundle        = bundles['SoftmaxBundle']

        network_layers_def = [
            InputBundle(shape=model.input_shape, noise=False),
            # Convolutional layers
            ConvBundle(num_filters=32, pool=True),

            ConvBundle(num_filters=32),
            ConvBundle(num_filters=32, pool=True),

            ConvBundle(num_filters=64),
            ConvBundle(num_filters=128, pool=True),

            ConvBundle(num_filters=256),
            ConvBundle(num_filters=256, pool=True),

            ConvBundle(num_filters=256),

            # Fully connected layers
            FullyConnectedBundle(num_units=128),
            FullyConnectedBundle(num_units=128),
            SoftmaxBundle(num_units=model.output_dims)
        ]
        network_layers = abstract_models.evaluate_layer_list(
            network_layers_def, verbose=verbose)
        #model.network_layers = network_layers
        output_layer = network_layers[-1]
        model.output_layer = output_layer
        return output_layer

    #def fit_interactive(X_train, y_train, X_valid, y_valid):
    #    pass


def get_shark_dataset(target_type='binary', data_type='chip'):
    """
    >>> from ibeis.scripts.classify_shark import *  # NOQA
    >>> target_type = 'binary'
    >>> data_type = 'hog'
    >>> dataset = get_shark_dataset(target_type)
    """
    from ibeis_cnn.dataset import DataSet
    from ibeis.scripts import classify_shark
    tup = classify_shark.get_shark_labels_and_metadata(target_type)
    ibs, annots, target, target_names, config, metadata, enc = tup
    data_shape = config['dim_size'] + (3,)
    nTotal = len(annots)

    # Build dataset configuration string
    trail_cfgstr = ibs.depc_annot.get_config_trail_str('chips', config)
    trail_hashstr = ut.hashstr27(trail_cfgstr)
    visual_uuids = annots.visual_uuids
    metadata['visual_uuid'] = np.array(visual_uuids)
    #metadata['nids'] = np.array(annots.nids)
    chips_hashstr = ut.hashstr_arr27(annots.visual_uuids, 'chips')
    cfgstr = chips_hashstr + '_' + trail_hashstr
    name = 'injur-shark'

    if data_type == 'hog':
        cfgstr = 'hog_' + cfgstr
        name += '-hog'

    training_dpath = ibs.get_neuralnet_dir()
    dataset = DataSet(cfgstr,
                      data_shape=data_shape,
                      num_data=nTotal,
                      training_dpath=training_dpath,
                      name=name)

    print(dataset.dataset_id)

    dataset.setprop('ibs', ibs)
    dataset.setprop('annots', annots)
    dataset.setprop('target_names', target_names)
    dataset.setprop('config', config)
    dataset.setprop('enc', enc)

    try:
        dataset.load()
    except IOError:
        import vtool as vt
        dataset.ensure_dirs()

        if data_type == 'hog':
            data = np.array([h.ravel() for h in annots.hog_hog])
            labels = target
            # Save data where dataset expects it to be
            dataset.save(data, labels, metadata, data_per_label=1)
        else:
            chip_gen = ibs.depc_annot.get('chips', annots.aids, 'img',
                                          eager=False, config=config)
            iter_ = iter(ut.ProgIter(chip_gen, nTotal=nTotal, lbl='load chip'))
            shape = (nTotal,) + data_shape
            data = vt.fromiter_nd(iter_, shape=shape, dtype=np.uint8)  # NOQA
            labels = target
            # Save data where dataset expects it to be
            dataset.save(data, labels, metadata, data_per_label=1)

    from ibeis_cnn.dataset import stratified_label_shuffle_split
    if not dataset.has_split('learn'):
        nids = np.array(dataset.metadata['nids'])
        # Partition into a testing and training dataset
        y = dataset.labels
        train_idx, test_idx = stratified_label_shuffle_split(
            y, nids, [.8, .2], rng=22019)
        nids_train = nids.take(train_idx, axis=0)
        y_train = y.take(train_idx, axis=0)
        # Partition training into learning and validation
        learn_idx, valid_idx = stratified_label_shuffle_split(
            y_train, nids_train,
            [.8, .2], idx=train_idx, rng=90120)
        assert len(np.intersect1d(learn_idx, test_idx)) == 0
        assert len(np.intersect1d(valid_idx, test_idx)) == 0
        assert len(np.intersect1d(learn_idx, valid_idx)) == 0
        if data_type == 'hog':
            dataset.add_split('train', train_idx)
        dataset.add_split('test', test_idx)
        dataset.add_split('learn', learn_idx)
        dataset.add_split('valid', valid_idx)
        dataset.clear_cache('full')

    if data_type == 'hog':
        # hack
        y = dataset.labels
        nids = np.array(dataset.metadata['nids'])
        train_idx, test_idx = stratified_label_shuffle_split(
            y, nids, [.8, .2], rng=22019)
        nids_train = nids.take(train_idx, axis=0)
        y_train = y.take(train_idx, axis=0)
        # Partition training into learning and validation
        learn_idx, valid_idx = stratified_label_shuffle_split(
            y_train, nids_train,
            [.8, .2], idx=train_idx, rng=90120)
        dataset._split_idxs = {}
        dataset._split_idxs['learn'] = learn_idx
        dataset._split_idxs['valid'] = valid_idx
        dataset._split_idxs['train'] = train_idx
        dataset._split_idxs['test'] = test_idx

    dataset.ensure_symlinked()
    return dataset


#def get_sharks_dataset(target_type=None, data_type='hog'):
#    """
#    Ignore:
#        # Binarize into multi-class labels
#        # http://stackoverflow.com/questions/10526579/use-scikit-learn-to-classify-into-multiple-categories
#        #menc = preprocessing.MultiLabelBinarizer()
#        #menc.fit(annot_tags)
#        #target = menc.transform(annot_tags)
#        #enc = menc
#        # henc = preprocessing.OneHotEncoder()
#        # henc.fit(menc.transform(annot_tags))
#        # target = henc.transform(menc.transform(annot_tags))
#        # target = np.array([int('healthy' not in tags) for tags in annots.case_tags])

#    CommandLine:
#        python -m ibeis.scripts.classify_shark get_sharks_dataset
#        python -m ibeis.scripts.classify_shark get_sharks_dataset --show --monitor

#    Example:
#        >>> target_type = 'binary'
#        >>> data_type = 'chip'
#        >>> from ibeis.scripts.classify_shark import *  # NOQA
#        >>> get_sharks_dataset(target_type, data_type)
#    """

#    data = None
#    tup = get_shark_labels_and_metadata(target_type)
#    ibs, annots, target, target_names, config, metadata, enc = tup
#    data = np.array([h.ravel() for h in annots.hog_hog])
#    # Build scipy / scikit data standards
#    ds = sklearn.datasets.base.Bunch(
#        ibs=ibs,
#        #data=data,
#        name='sharks',
#        DESCR='injured-vs-healthy whale sharks',
#        target_names=target_names,
#        target_labels=enc.transform(target_names),
#        config=config,
#        target=target,
#        enc=enc,
#        data=data,
#        **metadata
#    )
#    return ds


def get_shark_labels_and_metadata(target_type=None, ibs=None, config=None):
    import ibeis
    if ibs is None:
        ibs = ibeis.opendb('WS_ALL')
    if config is None:
        config = {
            #'dim_size': (256, 256),
            'dim_size': (224, 224),
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
            (['primary', 'secondary', 'pose:novel'], None),
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
    elif target_type == 'multiclass1':
        regex_map = [
            ('injur-trunc', 'injur-trunc'),
            ('injur-scar', 'injur-scar'),
            ('injur-bite', 'injur-scar'),
            ('injur-nicks', 'injur-other'),
            ('healthy', 'healthy'),
            ('injur-.*', 'injur-other'),
        ]
    elif target_type == 'multiclass2':
        regex_map = [
            ('injur-trunc', 'injur-trunc'),
            ('healthy', 'healthy'),
            ('injur-.*', None),
        ]
    else:
        raise ValueError('Unknown target_type=%r' % (target_type,))
    tag_vocab = ut.flat_unique(*case_tags)
    alias_map = ut.build_alias_map(regex_map, tag_vocab)
    unmapped = list(set(tag_vocab) - set(alias_map.keys()))
    print('unmapped = %r' % (unmapped,))
    case_tags2 = ut.alias_tags(case_tags, alias_map)

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

    metadata = {
        'aids': np.array(annots.aids),
        'nids': np.array(annots.nids),
    }
    tup = ibs, annots, target, target_names, config, metadata, enc
    return tup


@ut.reloadable_class
class ClfProblem(object):
    """ Harness for researching a classification problem """
    def __init__(problem, ds):
        problem.ds = ds

    def print_support_info(problem):
        enc = problem.ds.enc
        target_labels = enc.inverse_transform(problem.ds.target)
        label_hist = ut.dict_hist(target_labels)
        print('support hist' + ut.repr3(label_hist))

    def fit_new_classifier(problem, train_idx):
        """
        References:
            http://leon.bottou.org/research/stochastic
            http://blog.explainmydata.com/2012/06/ntrain-24853-ntest-25147-ncorrupt.html
            http://scikit-learn.org/stable/modules/svm.html#svm-classification
            http://scikit-learn.org/stable/modules/grid_search.html
        """
        print('[problem] train classifier on %d data points' % (len(train_idx)))
        data = problem.ds.data
        target = problem.ds.target
        x_train = data.take(train_idx, axis=0)
        y_train = target.take(train_idx, axis=0)
        clf = sklearn.svm.SVC(kernel=str('linear'), C=.17, class_weight='balanced',
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

    def fit_new_linear_svm(problem, train_idx):
        print('[problem] train classifier on %d data points' % (len(train_idx)))
        data = problem.ds.data
        target = problem.ds.target
        x_train = data.take(train_idx, axis=0)
        y_train = target.take(train_idx, axis=0)
        clf = sklearn.svm.SVC(kernel=str('linear'), C=.17, class_weight='balanced',
                              decision_function_shape='ovr')
        clf.fit(x_train, y_train)

    def gridsearch_linear_svm_params(problem, train_idx):
        """
        Example:
            >>> from ibeis.scripts.classify_shark import *  # NOQA
            >>> from ibeis.scripts import classify_shark
            >>> ds = classify_shark.get_sharks_dataset('binary')
            >>> problem = classify_shark.ClfProblem(ds)
            >>> problem.print_support_info()
        """
        try:
            import sklearn.model_selection
        except ImportError:
            pass
        import sklearn.grid_search

        with ut.Timer('cv'):
            data = problem.ds.data
            target = problem.ds.target

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

            train_idx = stratified_sample_idxs_unbalanced(target, 4000)

            x_train = data.take(train_idx, axis=0)
            y_train = target.take(train_idx, axis=0)
            param_grid = {
                #'C': [1, .5, .1, 5, 10, 100],
                #'C': [1, 1e-1, 1e-2, 1e-3]
                #'C': [1, 1e-1, 1e-2, 1e-3]
                #'C': np.linspace(1, 1e-5, 15)
                #'C': np.linspace(.2, 1e-5, 15)
                #'C': np.logspace(np.log10(1e-3), np.log10(.1), 30, base=10)
                #'C': np.linspace(.1, .3, 20),
                #'C': np.linspace(1.0, .22, 20),
                'C': np.linspace(.25, .01, 40),
                #'loss': ['l2', 'l1'],
                #'penalty': ['l2', 'l1'],
            }
            _clf = sklearn.svm.SVC(kernel=str('linear'), C=.17, class_weight='balanced',
                                   decision_function_shape='ovr')
            clf = sklearn.grid_search.GridSearchCV(_clf, param_grid, n_jobs=6,
                                                   iid=False, cv=5, verbose=3)
            clf.fit(x_train, y_train)
            print('clf.best_params_ = %r' % (clf.best_params_,))
            print("Best parameters set found on development set:")
            print(clf.best_params_)
            print("Grid scores on development set:")
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() * 2, params))
            xdata = np.array([t[0]['C'] for t in clf.grid_scores_])
            ydata = np.array([t[1] for t in clf.grid_scores_])

            y_data_std = np.array([t[2].std() for t in clf.grid_scores_])
            ydata_mean = ydata
            y_data_max = ydata_mean + y_data_std
            y_data_min = ydata_mean - y_data_std

            #pt.plot(xdata, ydata, '-rx')
            import plottool as pt
            pt.figure(fnum=pt.ensure_fnum(None))
            ax = pt.gca()
            ax.fill_between(xdata, y_data_min, y_data_max, alpha=.2, color=pt.LIGHT_BLUE)
            pt.draw_hist_subbin_maxima(ydata, xdata)

            #clf.best_params_ = {u'C': 0.07143785714285722}
            #Best parameters set found on development set:
            #{u'C': 0.07143785714285722}
            #Grid scores on development set:
            #0.729 (+/-0.016) for {u'C': 1.0}
            #0.729 (+/-0.019) for {u'C': 0.92857214285714285}
            #0.733 (+/-0.017) for {u'C': 0.85714428571428569}
            #0.734 (+/-0.015) for {u'C': 0.78571642857142865}
            #0.736 (+/-0.016) for {u'C': 0.71428857142857138}
            #0.739 (+/-0.020) for {u'C': 0.64286071428571434}
            #0.742 (+/-0.020) for {u'C': 0.57143285714285719}
            #0.743 (+/-0.021) for {u'C': 0.50000500000000003}
            #0.746 (+/-0.023) for {u'C': 0.42857714285714288}
            #0.749 (+/-0.023) for {u'C': 0.35714928571428572}
            #0.755 (+/-0.025) for {u'C': 0.28572142857142857}
            #0.760 (+/-0.027) for {u'C': 0.21429357142857142}
            #0.762 (+/-0.025) for {u'C': 0.14286571428571437}
            #0.770 (+/-0.036) for {u'C': 0.07143785714285722}
            #0.664 (+/-0.031) for {u'C': 1.0000000000000001e-05}

            #0.774 (+/-0.039) for {u'C': 0.017433288221999882}
            #0.775 (+/-0.039) for {u'C': 0.020433597178569417}
            #0.774 (+/-0.039) for {u'C': 0.023950266199874861}
            #0.777 (+/-0.038) for {u'C': 0.02807216203941177}
            #0.775 (+/-0.036) for {u'C': 0.032903445623126679}
            #0.773 (+/-0.033) for {u'C': 0.038566204211634723}

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
        y = problem.ds.target
        rng = 43432
        if hasattr(problem.ds, 'nids'):
            # Ensure that an individual does not appear in both the train
            # and the test dataset
            from ibeis_cnn.dataset import stratified_kfold_label_split
            labels = problem.ds.nids
            _iter = stratified_kfold_label_split(y, labels, n_folds=n_folds, rng=rng)
        else:
            xvalkw = dict(n_folds=n_folds, shuffle=True, random_state=rng)
            import sklearn.cross_validation
            skf = sklearn.cross_validation.StratifiedKFold(y, **xvalkw)
            _iter = skf
            #import sklearn.model_selection
            #skf = sklearn.model_selection.StratifiedKFold(**xvalkw)
            #_iter = skf.split(X=np.empty(len(y)), y=y)
        msg = 'cross-val test on %s' % (problem.ds.name)
        progiter = ut.ProgIter(_iter, nTotal=n_folds, lbl=msg)
        for train_idx, test_idx in progiter:
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

    def print_report(result):
        report = sklearn.metrics.classification_report(
            result.y_true, result.y_pred,
            target_names=result.ds.target_names)
        print(report)


def shark_svm():
    r"""
    References:
        http://scikit-learn.org/stable/model_selection.html

    TODO:
        * Change unreviewed healthy tags to healthy-likely

    CommandLine:
        python -m ibeis.scripts.classify_shark shark_svm --show

    Example:
        >>> from ibeis.scripts.classify_shark import *  # NOQA
        >>> shark_svm()
        >>> ut.show_if_requested()
    """
    from ibeis.scripts import classify_shark
    import sklearn.metrics
    import plottool as pt
    import pandas as pd
    pd.set_option("display.max_rows", 20)
    pt.qt4ensure()

    #target_type = 'binary'
    target_type = 'multiclass1'
    #target_type = 'multiclass2'
    #dataset = classify_shark.get_shark_dataset(target_type)

    ds = classify_shark.get_shark_dataset(target_type, 'hog')
    # Make resemble old dataset
    # FIXME; make ibeis_cnn dataset work here too
    #annots = ds.getprop('annots')
    ds.enc = ds.getprop('enc')
    ds.aids = ds.getprop('annots').aids
    ds.target = ds.labels
    ds.target_names = ds.getprop('target_names')
    ds.target_labels = ds.enc.transform(ds.target_names)
    ds.ibs = ds.getprop('ibs')
    ds.config = ds.getprop('config')

    problem = classify_shark.ClfProblem(ds)
    problem.print_support_info()

    result_list = []
    #train_idx, test_idx = problem.stratified_2sample_idxs()

    train_idx = ds._split_idxs['train']
    test_idx = ds._split_idxs['test']

    clf = problem.fit_new_classifier(train_idx)
    result = problem.test_classifier(clf, test_idx)
    result_list.append(result)

    #n_folds = 10
    #for train_idx, test_idx in problem.gen_crossval_idxs(n_folds):
    #    clf = problem.fit_new_classifier(train_idx)
    #    result = problem.test_classifier(clf, test_idx)
    #    result_list.append(result)

    for result in result_list:
        result.compile_results()

    for result in result_list:
        result.print_report()

    isect_sets = [set(s1).intersection(set(s2)) for s1, s2 in ut.combinations([
        result.df.index for result in result_list], 2)]
    assert all([len(s) == 0 for s in isect_sets]), ('cv sets should not intersect')

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
    print(pd.DataFrame(confusion, columns=[m for m in result.ds.target_names],
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
        sl = snapped_slice(len(df_chunk), frac, n)
        idx = df_chunk.index[sl]
        df_chunk = df_chunk.loc[idx]
        place_name = 'hardness=%.2f' % (frac,)
        if target is not None:
            df_chunk.nice = place_name + ' ' + ds.target_names[target]
        else:
            df_chunk.nice = place_name
        return df_chunk

    n = 4
    fracs = [0.0, .7, .8, .9, 1.0]
    df_chunks = [grab_subchunk(frac, n, target)
                 for frac in fracs for target in ds.target_labels]

    ibs = ds.ibs
    config = ds.config
    from ibeis_cnn import draw_results
    inter = draw_results.make_InteractClasses(ibs, config, df_chunks,
                                              nCols=len(ds.target_labels))
    inter.start()

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
