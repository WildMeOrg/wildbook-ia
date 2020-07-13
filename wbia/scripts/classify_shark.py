# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.svm
import sklearn.metrics
from sklearn import preprocessing

try:
    from wbia_cnn.models import abstract_models

    AbstractCategoricalModel = abstract_models.AbstractCategoricalModel
except ImportError:
    AbstractCategoricalModel = object
    print('no wbia_cnn')

from os.path import join

(print, rrr, profile) = ut.inject2(__name__)


def shark_net(dry=False):
    """
    CommandLine:
        python -m wbia.scripts.classify_shark shark_net
        python -m wbia.scripts.classify_shark shark_net --dry
        python -m wbia.scripts.classify_shark shark_net --vd --monitor

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.classify_shark import *  # NOQA
        >>> shark_net()
    """
    from wbia.scripts import classify_shark
    import wbia

    ibs = wbia.opendb('WS_ALL')  # NOQA
    config = {'dim_size': (224, 224), 'resize_dim': 'wh'}

    # ------------
    # Define dataset
    # ------------
    target_type = 'binary'
    # target_type = 'multiclass3'
    # ut.delete(ibs.get_neuralnet_dir())  # to reset
    dataset = classify_shark.get_shark_dataset(target_type, 'chip')

    # ------------
    # Define model
    # ------------
    if ut.get_computer_name() == 'Leviathan':
        batch_size = 128
        suffix = 'resnet'
        # suffix = 'lenet'
        # suffix = 'incep'
    else:
        suffix = 'lenet'
        batch_size = 64
        # suffix = 'resnet'
        # batch_size = 32

    model_name = 'injur-shark-' + suffix

    if False:
        model = classify_shark.WhaleSharkInjuryModel(
            name=model_name,
            output_dims=len(dataset.getprop('target_names')),
            data_shape=config['dim_size'] + (3,),
            batch_size=batch_size,
            arch_dpath='.',
        )
        model.init_arch()
        model.load_model_state()
    else:
        model = classify_shark.WhaleSharkInjuryModel(
            name=model_name,
            dataset_dpath=dataset.dataset_dpath,
            training_dpath=ibs.get_neuralnet_dir(),
            #
            output_dims=len(dataset.getprop('target_names')),
            data_shape=config['dim_size'] + (3,),
            batch_size=batch_size,
        )
        model.init_arch()
        model.print_layer_info()

    if False:
        model.arch_dpath = '/home/joncrall/Desktop/manually_saved/arch_injur-shark-resnet_o2_d27_c2942_jzuddodd/'

        state_fpath = model.get_model_state_fpath(dpath=model.trained_arch_dpath)
        state_fpath = model.get_model_state_fpath()
        model.load_model_state(fpath=state_fpath)

        # X_test, y_test = dataset.subset('test')
        # X_test, y_test = dataset.subset('valid')
        # X_test, y_test = dataset.subset('learn')
        X_test, y_test = dataset.subset('test')
        # y_pred = model.predict(X_test)
        test_outptuts = model._predict(X_test)
        y_pred = test_outptuts['predictions']
        print(model.name)
        report = sklearn.metrics.classification_report(y_true=y_test, y_pred=y_pred,)
        print(report)

        state_fpath = '/home/joncrall/Desktop/manually_saved/arch_injur-shark-resnet_o2_d27_c2942_jzuddodd/model_state_arch_jzuddodd.pkl'
        dpath = '/home/joncrall/Desktop/manually_saved/arch_injur-shark-lenet_o2_d11_c688_acioqbst'
        model.dump_cases(X_test, y_test, 'test', dpath=dpath)

    hyperparams = dict(
        era_size=30,
        max_epochs=1000,
        rate_schedule=0.1,
        augment_on=True,
        class_weight='balanced',
        stopping_patience=200,
    )
    model.learn_state.weight_decay = 0.000002
    model.learn_state.learning_rate = 0.005
    ut.update_existing(model.hyperparams, hyperparams, assert_exists=True)
    model.monitor_config['monitor'] = True
    model.monitor_config['weight_dump_freq'] = 100
    model.monitor_config['case_dump_freq'] = 100

    # model.build_backprop_func()
    # model.build_forward_func()

    # ---------------
    # Setup and learn
    # ---------------

    X_learn, y_learn = dataset.subset('learn')
    X_valid, y_valid = dataset.subset('valid')
    X_test, y_test = dataset.subset('test')
    # model.ensure_data_params(X_learn, y_learn)
    # X_train = X_learn  # NOQA
    # y_train = y_learn  # NOQA
    valid_idx = None  # NOQA

    if dry or ut.get_argflag('--dry'):
        return model, dataset
    model.fit(X_learn, y_learn, X_valid=X_valid, y_valid=y_valid)


# @ut.reloadable_class
class WhaleSharkInjuryModel(AbstractCategoricalModel):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.classify_shark import *  # NOQA
        >>> from wbia.scripts import classify_shark
        >>> ds = classify_shark.get_sharks_dataset('binary', 'chip')
        >>> problem = classify_shark.ClfProblem(ds)
        >>> problem.print_support_info()
        >>> ibs = ds.ibs
    """

    def def_lenet(model):
        import wbia_cnn.__LASAGNE__ as lasange
        from wbia_cnn import custom_layers

        print('[model] init_arch')

        lrelu = lasange.nonlinearities.LeakyRectify(leakiness=(1.0 / 3.0))
        W = lasange.init.Orthogonal('relu')

        bundles = custom_layers.make_bundles(
            nonlinearity=lrelu,
            batch_norm=True,
            filter_size=(3, 3),
            stride=(1, 1),
            pool_size=(2, 2),
            pool_stride=(2, 2),
            W=W,
        )
        b = ut.DynStruct(copy_dict=bundles)

        network_layers_def = [
            b.InputBundle(shape=model.input_shape, noise=False),
            # Convolutional layers
            b.ConvBundle(num_filters=16, pool=True),
            b.ConvBundle(num_filters=16),
            b.ConvBundle(num_filters=16, pool=True),
            b.ConvBundle(num_filters=16),
            b.ConvBundle(num_filters=32, pool=True),
            b.ConvBundle(num_filters=32),
            b.ConvBundle(num_filters=32, pool=True),
            b.ConvBundle(num_filters=32),
            # Fully connected layers
            b.DenseBundle(num_units=64, dropout=0.5),
            b.DenseBundle(num_units=64, dropout=0.5),
            b.SoftmaxBundle(num_units=model.output_dims),
        ]
        return network_layers_def

    def def_resnet(model):
        import wbia_cnn.__LASAGNE__ as lasange
        from wbia_cnn import custom_layers

        print('[model] init_arch')
        nonlinearity = lasange.nonlinearities.LeakyRectify(leakiness=(1.0 / 3.0))
        W = lasange.init.HeNormal(gain='relu')
        # W = lasange.init.GlorotUniform()

        bundles = custom_layers.make_bundles(
            nonlinearity=nonlinearity,
            filter_size=(3, 3),
            stride=(1, 1),
            W=W,
            pool_size=(2, 2),
            pool_stride=(2, 2),
        )
        b = ut.DynStruct(copy_dict=bundles)

        network_layers_def = [
            b.InputBundle(shape=model.input_shape, noise=False),
            # Convolutional layers
            b.ConvBundle(num_filters=16, pool=False),
            b.ResidualBundle(num_filters=16, stride=(2, 2), preactivate=False),
            b.ResidualBundle(num_filters=16),
            b.ResidualBundle(num_filters=16, stride=(2, 2)),
            b.ResidualBundle(num_filters=16),
            b.ResidualBundle(num_filters=16, stride=(2, 2)),
            b.ResidualBundle(num_filters=16),
            b.ResidualBundle(num_filters=16, stride=(2, 2)),
            b.ResidualBundle(num_filters=16, dropout=None),
            b.ResidualBundle(num_filters=16, stride=(2, 2), dropout=0.5),
            b.ResidualBundle(num_filters=16, postactivate=True, dropout=0.5),
            # Fully connected layers
            b.GlobalPool(),
            b.SoftmaxBundle(num_units=model.output_dims),
        ]
        return network_layers_def

    def def_inception(model):
        import wbia_cnn.__LASAGNE__ as lasange
        from wbia_cnn import custom_layers

        print('[model] init_arch')

        N = 16

        # Define default incption branch types
        incep_branches = [
            dict(t='c', s=(1, 1), r=0, n=N),
            dict(t='c', s=(3, 3), r=N // 2, n=N // 2),
            dict(t='c', s=(3, 3), r=N // 4, n=N // 4, d=2),
            dict(t='p', s=(3, 3), n=N // 2),
        ]

        lrelu = lasange.nonlinearities.LeakyRectify(leakiness=(1.0 / 3.0))
        W = lasange.init.Orthogonal('relu')

        bundles = custom_layers.make_bundles(
            nonlinearity=lrelu,
            batch_norm=True,
            filter_size=(3, 3),
            stride=(1, 1),
            pool_size=(3, 3),
            pool_stride=(2, 2),
            branches=incep_branches,
            W=W,
        )
        b = ut.DynStruct(copy_dict=bundles)

        network_layers_def = [
            # Convolutional layers
            b.InputBundle(shape=model.input_shape, noise=False),
            b.ConvBundle(num_filters=16, filter_size=(3, 3), pool=False),
            b.ConvBundle(num_filters=12, filter_size=(3, 3), pool=True),
            b.InceptionBundle(dropout=0.3, pool=True),
            b.InceptionBundle(dropout=0.3, pool=True),
            b.InceptionBundle(dropout=0.4, pool=True),
            b.InceptionBundle(
                dropout=0.5,
                branches=[
                    dict(t='c', s=(1, 1), r=0, n=model.output_dims),
                    dict(t='c', s=(3, 3), r=N // 2, n=model.output_dims),
                    dict(t='c', s=(3, 3), r=N // 4, n=model.output_dims, d=2),
                    dict(t='p', s=(3, 3), n=model.output_dims),
                ],
            ),
            b.GlobalPool(),
            b.SoftmaxBundle(num_units=model.output_dims)
            # Fully connected layers
            # b.DenseBundle(num_units=64, dropout=.5),
            # b.DenseBundle(num_units=64, dropout=.5),
        ]
        return network_layers_def

    def init_arch(model, verbose=ut.VERBOSE, **kwargs):
        r"""

        CommandLine:
            python -m wbia.scripts.classify_shark WhaleSharkInjuryModel.init_arch
            python -m wbia.scripts.classify_shark WhaleSharkInjuryModel.init_arch --show

            python -m wbia.scripts.classify_shark shark_net --dry --show
            python -m wbia.scripts.classify_shark shark_net --vd

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.classify_shark import *  # NOQA
            >>> verbose = True
            >>> data_shape = tuple(ut.get_argval('--datashape', type_=list,
            >>>                                  default=(224, 224, 3)))
            >>> model = WhaleSharkInjuryModel(batch_size=64, output_dims=2,
            >>>                               data_shape=data_shape)
            >>> model.init_arch()
            >>> model.print_model_info_str()
            >>> ut.quit_if_noshow()
            >>> model.show_arch(fullinfo=False)
            >>> ut.show_if_requested()
        """
        from wbia_cnn import custom_layers

        # if ut.get_computer_name() == 'Leviathan':
        if model.name.endswith('incep'):
            network_layers_def = model.def_inception()
        elif model.name.endswith('lenet'):
            network_layers_def = model.def_lenet()
        elif model.name.endswith('resnet'):
            network_layers_def = model.def_resnet()
        network_layers = custom_layers.evaluate_layer_list(network_layers_def)
        # model.network_layers = network_layers
        output_layer = network_layers[-1]
        model.output_layer = output_layer
        return output_layer

    def special_output():
        pass

    # def special_loss_function(output_activations):
    #    output_injur1 = output_activations[:, 0]
    #    output_injur2 = output_activations[:, 1]
    #    output_healthy = (1 - ((1 - output_injur1) * (1 - output_injur2))
    #    import wbia_cnn.__LASAGNE__ as lasange
    #    lasange.objectives.binary_crossentropy(output_injur1)
    #    lasange.objectives.binary_crossentropy(output_injur2)

    def augment(self, Xb, yb=None):
        """
        X_valid, y_valid = dataset.subset('valid')
        num = 10
        Xb = X_valid[:num]
        Xb = Xb / 255.0 if ut.is_int(Xb) else Xb
        Xb = Xb.astype(np.float32, copy=True)
        yb = None if yb is None else yb.astype(np.int32, copy=True)
        # Rescale the batch data to the range 0 to 1
        Xb_, yb_ = model.augment(Xb)
        yb_ = None
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> pt.qt4ensure()
        >>> from wbia_cnn import augment
        >>> augment.show_augmented_patches(Xb, Xb_, yb, yb_, data_per_label=1)
        >>> ut.show_if_requested()
        """
        from wbia_cnn import augment

        rng = np.random
        affperterb_ranges = dict(
            zoom_range=(1.3, 1.2),
            max_tx=2,
            max_ty=2,
            max_shear=ut.TAU / 32,
            max_theta=ut.TAU,
            enable_stretch=True,
            enable_flip=True,
        )
        Xb_, yb_ = augment.augment_affine(
            Xb,
            yb,
            rng=rng,
            inplace=True,
            data_per_label=1,
            affperterb_ranges=affperterb_ranges,
            aug_prop=0.5,
        )
        return Xb_, yb_

    # def fit_interactive(X_train, y_train, X_valid, y_valid):
    #    pass


def get_shark_dataset(target_type='binary', data_type='chip'):
    """
    >>> from wbia.scripts.classify_shark import *  # NOQA
    >>> target_type = 'binary'
    >>> data_type = 'hog'
    >>> dataset = get_shark_dataset(target_type)
    """
    from wbia_cnn.dataset import DataSet
    from wbia.scripts import classify_shark

    tup = classify_shark.get_shark_labels_and_metadata(target_type)
    ibs, annots, target, target_names, config, metadata, enc = tup
    data_shape = config['dim_size'] + (3,)
    length = len(annots)

    # Build dataset configuration string
    trail_cfgstr = ibs.depc_annot.get_config_trail_str('chips', config)
    trail_hashstr = ut.hashstr27(trail_cfgstr)
    visual_uuids = annots.visual_uuids
    metadata['visual_uuid'] = np.array(visual_uuids)
    # metadata['nids'] = np.array(annots.nids)
    chips_hashstr = ut.hashstr_arr27(annots.visual_uuids, 'chips')
    cfgstr = chips_hashstr + '_' + trail_hashstr
    name = 'injur-shark'

    if data_type == 'hog':
        cfgstr = 'hog_' + cfgstr
        name += '-hog'

    training_dpath = ibs.get_neuralnet_dir()
    dataset = DataSet(
        cfgstr,
        data_shape=data_shape,
        num_data=length,
        training_dpath=training_dpath,
        name=name,
    )

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
            chip_gen = ibs.depc_annot.get(
                'chips', annots.aids, 'img', eager=False, config=config
            )
            iter_ = iter(ut.ProgIter(chip_gen, length=length, lbl='load chip'))
            shape = (length,) + data_shape
            data = vt.fromiter_nd(iter_, shape=shape, dtype=np.uint8)  # NOQA
            labels = target
            # Save data where dataset expects it to be
            dataset.save(data, labels, metadata, data_per_label=1)

    from wbia_cnn.dataset import stratified_label_shuffle_split

    if not dataset.has_split('learn'):
        nids = np.array(dataset.metadata['nids'])
        # Partition into a testing and training dataset
        y = dataset.labels
        train_idx, test_idx = stratified_label_shuffle_split(
            y, nids, [0.8, 0.2], rng=22019
        )
        nids_train = nids.take(train_idx, axis=0)
        y_train = y.take(train_idx, axis=0)
        # Partition training into learning and validation
        learn_idx, valid_idx = stratified_label_shuffle_split(
            y_train, nids_train, [0.8, 0.2], y_idx=train_idx, rng=90120
        )
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
            y, nids, [0.8, 0.2], rng=22019
        )
        nids_train = nids.take(train_idx, axis=0)
        y_train = y.take(train_idx, axis=0)
        # Partition training into learning and validation
        learn_idx, valid_idx = stratified_label_shuffle_split(
            y_train, nids_train, [0.8, 0.2], y_idx=train_idx, rng=90120
        )
        dataset._split_idxs = {}
        dataset._split_idxs['learn'] = learn_idx
        dataset._split_idxs['valid'] = valid_idx
        dataset._split_idxs['train'] = train_idx
        dataset._split_idxs['test'] = test_idx

    dataset.ensure_symlinked()
    return dataset


def get_shark_labels_and_metadata(target_type=None, ibs=None, config=None):
    """
    >>> from wbia.scripts.classify_shark import *  # NOQA
    >>> target_type = 'multiclass3'
    >>> data_type = 'hog'
    """
    import wbia

    if ibs is None:
        ibs = wbia.opendb('WS_ALL')
    if config is None:
        config = {
            # 'dim_size': (256, 256),
            'dim_size': (224, 224),
            'resize_dim': 'wh',
        }
    all_annots = ibs.annots(config=config)

    isempty = ut.not_list(ut.lmap(len, ibs.images().aids))
    # if False:
    #    x = ibs.images().compress(isempty)
    num_empty_images = sum(isempty)
    print('Images without annotations: %r' % (num_empty_images,))

    print(
        'Building labels for %r annotations from %r images'
        % (len(all_annots), len(ut.unique(all_annots.gids)))
    )

    TARGET_TYPE = 'binary'
    # TARGET_TYPE = 'multiclass3'
    if target_type is None:
        target_type = TARGET_TYPE

    from wbia.scripts import getshark

    category_tags = getshark.get_injur_categories(all_annots)
    print('Base Category Tags tags')
    print(ut.repr3(ut.dict_hist(ut.flatten(category_tags))))

    print('Base Co-Occurrence Freq')
    co_occur1 = ut.tag_coocurrence(category_tags)
    print(ut.repr3(co_occur1))

    ntags_list = np.array(ut.lmap(len, category_tags))
    is_no_tag = ntags_list == 0
    is_single_tag = ntags_list == 1
    is_multi_tag = ntags_list > 1

    if target_type == 'binary':
        regex_map = [
            ('injur-.*', 'injured'),
            ('healthy', 'healthy'),
        ]
    elif target_type == 'multiclass3':
        regex_map = [
            ('injur-trunc', 'injur-trunc'),
            ('injur-nicks', 'injur-trunc'),
            ('injur-scar', 'injur-scar'),
            ('injur-bite', 'injur-scar'),
            ('injur-gill', 'injur-scar'),
            ('injur-other', None),
            ('injur-dead', None),
            ('healthy', 'healthy'),
        ]
    elif target_type == 'multiclassX':
        regex_map = [
            ('injur-trunc', 'injur-trunc'),
            ('healthy', 'healthy'),
            ('injur-.*', None),
        ]
    else:
        raise ValueError('Unknown target_type=%r' % (target_type,))

    tag_vocab = ut.flat_unique(*category_tags)
    alias_map = ut.build_alias_map(regex_map, tag_vocab)
    unmapped = list(set(tag_vocab) - set(alias_map.keys()))
    print('unmapped = %r' % (unmapped,))
    category_tags2 = ut.alias_tags(category_tags, alias_map)

    ntags_list = np.array(ut.lmap(len, category_tags2))
    is_no_tag = ntags_list == 0
    is_single_tag = ntags_list == 1
    is_multi_tag = ntags_list > 1

    print('Cleaned tags')
    hist = ut.tag_hist(category_tags2)
    print(ut.repr3(hist))

    # Get tag co-occurrence
    print('Co-Occurrence Freq')
    co_occur = ut.tag_coocurrence(category_tags2)
    print(ut.repr3(co_occur))

    print('Co-Occurrence Percent')
    co_occur_percent = ut.odict(
        [(keys, [100 * val / hist[k] for k in keys]) for keys, val in co_occur.items()]
    )
    print(ut.repr3(co_occur_percent, precision=2, nl=1))

    multi_annots = all_annots.compress(is_multi_tag)  # NOQA
    # ibs.set_image_imagesettext(multi_annots.gids, ['MultiTaged'] * is_multi_tag.sum())

    print("can't use %r annots due to no labels" % (is_no_tag.sum(),))
    print("can't use %r annots due to inconsistent labels" % (is_multi_tag.sum(),))
    print('will use %r annots with consistent labels' % (is_single_tag.sum(),))

    annot_tags = ut.compress(category_tags2, is_single_tag)
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


# @ut.reloadable_class
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
        clf = sklearn.svm.SVC(
            kernel=str('linear'),
            C=0.17,
            class_weight='balanced',
            decision_function_shape='ovr',
        )

        # C, penalty, loss
        # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        #              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        #              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        # clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
        # clf = clf.fit(X_train_pca, y_train)
        clf.fit(x_train, y_train)
        return clf

    def fit_new_linear_svm(problem, train_idx):
        print('[problem] train classifier on %d data points' % (len(train_idx)))
        data = problem.ds.data
        target = problem.ds.target
        x_train = data.take(train_idx, axis=0)
        y_train = target.take(train_idx, axis=0)
        clf = sklearn.svm.SVC(
            kernel=str('linear'),
            C=0.17,
            class_weight='balanced',
            decision_function_shape='ovr',
        )
        clf.fit(x_train, y_train)

    def gridsearch_linear_svm_params(problem, train_idx):
        """
        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.scripts.classify_shark import *  # NOQA
            >>> from wbia.scripts import classify_shark
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
                # 'C': [1, .5, .1, 5, 10, 100],
                # 'C': [1, 1e-1, 1e-2, 1e-3]
                # 'C': [1, 1e-1, 1e-2, 1e-3]
                # 'C': np.linspace(1, 1e-5, 15)
                # 'C': np.linspace(.2, 1e-5, 15)
                # 'C': np.logspace(np.log10(1e-3), np.log10(.1), 30, base=10)
                # 'C': np.linspace(.1, .3, 20),
                # 'C': np.linspace(1.0, .22, 20),
                'C': np.linspace(0.25, 0.01, 40),
                # 'loss': ['l2', 'l1'],
                # 'penalty': ['l2', 'l1'],
            }
            _clf = sklearn.svm.SVC(
                kernel=str('linear'),
                C=0.17,
                class_weight='balanced',
                decision_function_shape='ovr',
            )
            clf = sklearn.grid_search.GridSearchCV(
                _clf, param_grid, n_jobs=6, iid=False, cv=5, verbose=3
            )
            clf.fit(x_train, y_train)
            # (NOTE grid.predict only uses the best estimator)
            print('clf.best_params_ = %r' % (clf.best_params_,))
            print('Best parameters set found on development set:')
            print(clf.best_params_)
            print('Grid scores on development set:')
            for params, mean_score, scores in clf.grid_scores_:
                print('%0.3f (+/-%0.03f) for %r' % (mean_score, scores.std() * 2, params))
            xdata = np.array([t[0]['C'] for t in clf.grid_scores_])
            ydata = np.array([t[1] for t in clf.grid_scores_])

            y_data_std = np.array([t[2].std() for t in clf.grid_scores_])
            ydata_mean = ydata
            y_data_max = ydata_mean + y_data_std
            y_data_min = ydata_mean - y_data_std

            # pt.plot(xdata, ydata, '-rx')
            import wbia.plottool as pt

            pt.figure(fnum=pt.ensure_fnum(None))
            ax = pt.gca()
            ax.fill_between(xdata, y_data_min, y_data_max, alpha=0.2, color=pt.LIGHT_BLUE)
            pt.draw_hist_subbin_maxima(ydata, xdata)

            # y_data_std = np.array([t[2].std() for t in grid.grid_scores_])
            # ydata_mean = c_ydata
            # y_data_max = ydata_mean + y_data_std
            # y_data_min = ydata_mean - y_data_std
            # # import wbia.plottool as pt
            # # pt.figure(fnum=pt.ensure_fnum(None))
            # ax = pt.gca()
            # ax.fill_between(c_xdata, c_ydata, y_data_max, alpha=.2, color=pt.LIGHT_BLUE)
            # ax.fill_between(c_xdata, c_ydata, y_data_min, alpha=.2, color=pt.LIGHT_BLUE)
            # # pt.figure(fnum=pt.ensure_fnum(None))
            # hist = c_ydata
            # centers = c_xdata
            # pt.draw_hist_subbin_maxima(c_ydata, c_xdata, maxima_thresh=None, remove_endpoints=False)

            # clf.best_params_ = {u'C': 0.07143785714285722}
            # Best parameters set found on development set:
            # {u'C': 0.07143785714285722}
            # Grid scores on development set:
            # 0.729 (+/-0.016) for {u'C': 1.0}
            # 0.729 (+/-0.019) for {u'C': 0.92857214285714285}
            # 0.733 (+/-0.017) for {u'C': 0.85714428571428569}
            # 0.734 (+/-0.015) for {u'C': 0.78571642857142865}
            # 0.736 (+/-0.016) for {u'C': 0.71428857142857138}
            # 0.739 (+/-0.020) for {u'C': 0.64286071428571434}
            # 0.742 (+/-0.020) for {u'C': 0.57143285714285719}
            # 0.743 (+/-0.021) for {u'C': 0.50000500000000003}
            # 0.746 (+/-0.023) for {u'C': 0.42857714285714288}
            # 0.749 (+/-0.023) for {u'C': 0.35714928571428572}
            # 0.755 (+/-0.025) for {u'C': 0.28572142857142857}
            # 0.760 (+/-0.027) for {u'C': 0.21429357142857142}
            # 0.762 (+/-0.025) for {u'C': 0.14286571428571437}
            # 0.770 (+/-0.036) for {u'C': 0.07143785714285722}
            # 0.664 (+/-0.031) for {u'C': 1.0000000000000001e-05}

            # 0.774 (+/-0.039) for {u'C': 0.017433288221999882}
            # 0.775 (+/-0.039) for {u'C': 0.020433597178569417}
            # 0.774 (+/-0.039) for {u'C': 0.023950266199874861}
            # 0.777 (+/-0.038) for {u'C': 0.02807216203941177}
            # 0.775 (+/-0.036) for {u'C': 0.032903445623126679}
            # 0.773 (+/-0.033) for {u'C': 0.038566204211634723}

            # 0.722 (+/-0.060) for {u'C': 0.001}
            # 0.770 (+/-0.047) for {u'C': 0.01}
            # 0.775 (+/-0.047) for {u'C': 0.1}
            # 0.774 (+/-0.047) for {u'C': 0.12}
            # 0.773 (+/-0.045) for {u'C': 0.15}
            # 0.773 (+/-0.046) for {u'C': 0.17}
            # 0.772 (+/-0.047) for {u'C': 0.2}
            # 0.760 (+/-0.043) for {u'C': 0.5}
            # 0.748 (+/-0.043) for {u'C': 1.0}
            # 0.707 (+/-0.043) for {u'C': 100}
            # 0.702 (+/-0.047) for {u'C': 1000}

    def classifier_test(problem, clf, test_idx):
        print('[problem] test classifier on %d data points' % (len(test_idx),))
        data = problem.ds.data
        target = problem.ds.target
        X_test = data.take(test_idx, axis=0)
        y_true = target.take(test_idx, axis=0)

        y_conf = predict_svc_ovr(X_test)
        y_pred = y_conf.argmax(axis=1)

        result = ClfSingleResult(problem.ds, test_idx, y_true, y_pred, y_conf)
        return result

    def stratified_2sample_idxs(problem, frac=0.2, split_frac=0.75):
        target = problem.ds.target
        target_labels = problem.ds.target_labels

        rng = np.random.RandomState(43)
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
            from wbia_cnn.dataset import stratified_kfold_label_split

            labels = problem.ds.nids
            _iter = stratified_kfold_label_split(y, labels, n_folds=n_folds, rng=rng)
        else:
            xvalkw = dict(n_folds=n_folds, shuffle=True, random_state=rng)
            import sklearn.cross_validation

            skf = sklearn.cross_validation.StratifiedKFold(y, **xvalkw)
            _iter = skf
            # import sklearn.model_selection
            # skf = sklearn.model_selection.StratifiedKFold(**xvalkw)
            # _iter = skf.split(X=np.empty(len(y)), y=y)
        msg = 'cross-val test on %s' % (problem.ds.name)
        progiter = ut.ProgIter(_iter, length=n_folds, lbl=msg)
        for train_idx, test_idx in progiter:
            yield train_idx, test_idx


# @ut.reloadable_class
class ClfSingleResult(object):
    r"""
    Reports the results of a classification problem

    Example:
        >>> # DISABLE_DOCTEST
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
            result.y_true, result.y_pred, target_names=result.ds.target_names
        )
        print(report)


def get_model_state(clf):
    model_attr_names = [a for a in dir(clf) if a.endswith('_') and not a.startswith('__')]
    model_state = {a: getattr(clf, a) for a in model_attr_names}
    return model_state


def set_model_state(clf, model_state):
    attr_names = sorted(model_state.keys())
    attr_names1 = [
        'dual_coef_',
    ]
    attr_names2 = [
        'coef_',
    ]
    attr_names3 = attr_names1 + attr_names2
    attr_namesA = ut.isect(attr_names1, attr_names)
    attr_namesB = ut.setdiff(attr_names, attr_names3)
    attr_namesC = ut.isect(attr_names2, attr_names)
    attr_names_ = attr_namesA + attr_namesB + attr_namesC
    for a in attr_names_:
        val = model_state[a]
        print('a = %r' % (a,))
        try:
            setattr(clf, a, val)
        except AttributeError:
            val2 = getattr(clf, a)
            assert np.all(val == val2)


def predict_svc_ovr(clf, data):
    if len(clf.classes_) == 2:
        X = clf._validate_for_predict(data)
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
        y_conf = clf.decision_function(data)
    return y_conf


def predict_ws_injury_interim_svm(ibs, aids, **kwargs):
    """
    Returns relative confidence
    """
    config = {
        # 'dim_size': (256, 256),
        'dim_size': (224, 224),
        'resize_dim': 'wh',
    }

    # Load the SVM
    model_fname = 'interim_svc_injur-shark-hog_12559_224x224x3_ldhhxnxo.cPkl'
    model_url = 'https://wildbookiarepository.azureedge.net/models/{}'.format(model_fname)
    model_fpath = ut.grab_file_url(model_url, check_hash=False)
    clf = ut.load_cPkl(model_fpath)

    annots = ibs.annots(aids, config=config)
    data = np.array([h.ravel() for h in annots.hog_hog])

    target_names = ['healthy', 'injured']
    # confidence = clf.decision_function(data)
    # y_conf = predict_svc_ovr(clf, data)
    scores = clf.decision_function(data)
    y_pred = scores > 0.0
    y_pred = y_pred.astype(np.int64)
    # y_pred = clf.predict(data)
    ut.embed()
    pred_nice = ut.take(target_names, y_pred)
    return list(zip(pred_nice, scores))


def shark_svm():
    r"""
    References:
        http://scikit-learn.org/stable/model_selection.html

    TODO:
        * Change unreviewed healthy tags to healthy-likely

    CommandLine:
        python -m wbia.scripts.classify_shark shark_svm --show
        python -m wbia.scripts.classify_shark shark_svm

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.classify_shark import *  # NOQA
        >>> shark_svm()
        >>> ut.show_if_requested()
    """
    from wbia.scripts import classify_shark

    target_type = 'binary'
    # target_type = 'multiclass3'
    # dataset = classify_shark.get_shark_dataset(target_type)

    ds = classify_shark.get_shark_dataset(target_type, 'hog')
    # Make resemble old dataset
    # FIXME; make wbia_cnn dataset work here too
    # annots = ds.getprop('annots')
    ds.enc = ds.getprop('enc')
    ds.aids = ds.getprop('annots').aids
    ds.nids = ds.getprop('annots').nids
    ds.target = ds.labels
    ds.target_names = ds.getprop('target_names')
    ds.target_labels = ds.enc.transform(ds.target_names)
    ds.ibs = ds.getprop('ibs')
    ds.config = ds.getprop('config')

    problem = classify_shark.ClfProblem(ds)
    problem.print_support_info()

    # BUILD_RELEASE_MODEL = False
    # if BUILD_RELEASE_MODEL:
    #     clf = sklearn.svm.SVC(kernel=str('linear'), C=.17,
    #                           class_weight='balanced',
    #                           decision_function_shape='ovr',
    #                           verbose=10)
    #     clf.fit(ds.data, ds.target)
    #     model_fname = 'interim_svc_{}.cPkl'.format(ds.dataset_id)
    #     model_dpath = ut.ensuredir((ds.dataset_dpath, 'svms'))
    #     model_fpath = join(model_dpath, model_fname)
    #     ut.save_cPkl(model_fpath, clf)
    #     """
    #     TO PUBLISH
    #     scp clf to lev:/media/hdd/PUBLIC/models
    #     run script lev:/media/hdd/PUBLIC/hash.py to refresh hashes
    #     """
    #     user = ut.get_user_name()
    #     host = 'cthulhu.dyn.wildme.io'
    #     remote_path = '/data/public/models/' + model_fname
    #     remote_uri = user + '@' + host + ':' + remote_path
    #     ut.rsync(model_fpath, remote_uri)

    #     command = 'python /media/hdd/PUBLIC/hash.py'
    #     ut.cmd('ssh {user}@{host} "{command}"'.format(user=user, host=host,
    #                                                   command=command))

    model_dpath = ut.ensuredir((ds.dataset_dpath, 'svms'))
    # n_folds = 10
    n_folds = 10
    # ensemble_dpath = ut.ensuredir((model_dpath, 'svms_%d_fold' % (n_folds,)))

    train_idx = ds._split_idxs['train']
    test_idx = ds._split_idxs['test']

    y_train = ds.target.take(train_idx)
    nids_train = ut.take(ds.nids, train_idx)

    # Ensure that an individual does not appear in both train and test
    # _iter = stratified_kfold_label_split(y_train, nids_train, y_idx=train_idx,
    #                                     n_folds=n_folds, rng=rng)

    class MyLabelCV(object):
        def __init__(self, y_train, nids_train, n_folds):
            self.nids_train = nids_train
            self.y_train = y_train
            self.n_folds = n_folds

        def __len__(self):
            return self.n_folds

        def __iter__(self):
            from wbia_cnn.dataset import stratified_kfold_label_split

            rng = 1809629827
            for _ in stratified_kfold_label_split(
                self.y_train, self.nids_train, n_folds=self.n_folds, rng=rng
            ):
                yield _

    clf_fpath = join(model_dpath, '%s_svc_folds_%s.cPkl' % (target_type, n_folds))
    if not ut.checkpath(clf_fpath):
        """
        Curate strategy:
            Use gridsearch to select a reasonable C=.17
            Then train 10 classifiers with 10 split cross validation.
            This lets us make an "unbias" prediction for each training example.
            Look at predictions for all training examples (predict using only
                classifiers not trained with that point).
            Look at worst worst performing examples.
            Fix any errors that occur.
            Now that the database is better, we learn the actual model.

        Learning strategy:
            * Set aside a set test.
            * The remaining data is the training set.
            * Run Gridsearch with N-fold cross validation on training set to
              look at performance given different hyperparameters of the SVM.
            * Use quadratic interpolation to select a "best" parameter.

            (NOTE grid.predict only uses the best estimator (however it is a refit estimator))

            Train a single SVM using these parameters on all training data.

            Evaluate this SVM on the test set.
        """
        C = None
        if C is None:
            import sklearn
            import sklearn.grid_search
            import sklearn.svm

            # C controls the margin of the hyperplane.
            # Smaller C = Larger Hyperplane
            # So, the larger the C the less willing the SVM will be to get
            # examples wrong.

            param_grid = {
                # 'C': np.linspace(.1, .2, 10),
                'C': [
                    0.0001,
                    0.001,
                    0.005,
                    0.01,
                    0.08,
                    0.1,
                    0.12,
                    0.15,
                    0.17,
                    0.2,
                    0.22,
                    0.5,
                    1.0,
                    100,
                    1000,
                    10000,
                ]
                # 'C': np.linspace(.1, .2, 3),
            }
            clf = sklearn.svm.SVC(
                kernel=str('linear'),
                C=0.17,
                class_weight='balanced',
                decision_function_shape='ovr',
            )
            cv = MyLabelCV(y_train, nids_train, n_folds=n_folds)
            grid = sklearn.grid_search.GridSearchCV(
                clf,
                param_grid=param_grid,
                cv=cv,
                refit=False,
                n_jobs=min(n_folds, 6),
                verbose=10,
            )
            x_train = ds.data.take(train_idx, axis=0)
            y_train = ds.target.take(train_idx, axis=0)
            grid.fit(x_train, y_train)

            for params, mean_score, scores in grid.grid_scores_:
                print('%0.3f (+/-%0.03f) for %r' % (mean_score, scores.std() * 2, params))

            c_xdata = np.array([t[0]['C'] for t in grid.grid_scores_])
            c_ydata = np.array([t[1] for t in grid.grid_scores_])
            import vtool as vt

            # maxima_x, maxima_y, argmaxima = vt.hist_argmaxima(c_ydata, c_xdata, maxima_thresh=None)
            submaxima_x, submaxima_y = vt.argsubmaxima(c_ydata, c_xdata)
            # pt.draw_hist_subbin_maxima(c_ydata, c_xdata, maxima_thresh=None, remove_endpoints=False)
            C = submaxima_x[0]
            print('C = %r' % (C,))
        else:
            print('C = %r' % (C,))

        clf_all = sklearn.svm.SVC(
            kernel=str('linear'),
            C=C,
            class_weight='balanced',
            decision_function_shape='ovr',
            verbose=10,
        )
        X_train = ds.data.take(train_idx, axis=0)
        clf_all.fit(X_train, y_train)
        ut.save_data(clf_fpath, clf_all.__dict__)
        clf = clf_all
    else:
        clf = sklearn.svm.SVC()
        clf.__dict__.update(**ut.load_data(clf_fpath))

    def classifier_test(clf, X_test, y_test):
        print('[problem] test classifier on %d data points' % (len(test_idx),))
        y_conf = predict_svc_ovr(X_test)
        y_pred = y_conf.argmax(axis=1)
        result = ClfSingleResult(problem.ds, test_idx, y_test, y_pred, y_conf)
        return result

    test_idx = ds._split_idxs['test']
    X_test = ds.data.take(test_idx, axis=0)
    y_test = ds.target.take(test_idx, axis=0)
    result = classifier_test(clf, X_test, y_test)
    result.compile_results()
    result.print_report()

    result_list = [result]

    import pandas as pd

    # import wbia.plottool as pt
    # Combine information from results
    df = pd.concat([r.df for r in result_list])
    df['hardness'] = 1 / df['easiness']
    df['aid'] = ut.take(ds.aids, df.index)
    df['target'] = ut.take(ds.target, df.index)
    df['failed'] = df['pred'] != df['target']

    report = sklearn.metrics.classification_report(
        y_true=df['target'], y_pred=df['pred'], target_names=result.ds.target_names
    )
    print(report)

    confusion = sklearn.metrics.confusion_matrix(df['target'], df['pred'])
    print('Confusion Matrix:')
    print(
        pd.DataFrame(
            confusion,
            columns=[m for m in result.ds.target_names],
            index=['gt ' + m for m in result.ds.target_names],
        )
    )

    # inspect_results(ds, result_list)

    if False:
        if False:
            # train_idx, test_idx = problem.stratified_2sample_idxs()
            train_idx = ds._split_idxs['train']
            test_idx = ds._split_idxs['test']

            # import sklearn.metrics
            # model_dpath = join(ds.dataset_dpath, 'svms')
            # model_fpath = join(model_dpath, target_type + '_svc.cPkl')
            # if ut.checkpath(model_fpath):
            #    clf = sklearn.svm.SVC(kernel=str('linear'), C=.17, class_weight='balanced',
            #                          decision_function_shape='ovr')
            #    clf.__dict__.update(**ut.load_data(model_fpath))
            # else:
            #    clf = problem.fit_new_classifier(train_idx)
            #    ut.ensuredir(model_dpath)
            #    ut.save_data(model_fpath, clf.__dict__)
            result_list = []
            result = problem.test_classifier(clf, test_idx)
            result_list.append(result)

            for result in result_list:
                result.compile_results()

            for result in result_list:
                result.print_report()

            inspect_results(ds, result_list)
        if False:
            result_list = []
            result = problem.test_classifier(clf, train_idx)
            result_list.append(result)
            for result in result_list:
                result.compile_results()
            for result in result_list:
                result.print_report()
            inspect_results(ds, result_list)
        if False:
            result_list = []
            # View support vectors
            support_idxs = clf.support_
            result = problem.test_classifier(clf, support_idxs)
            result_list.append(result)
            for result in result_list:
                result.compile_results()
            for result in result_list:
                result.print_report()
            inspect_results(ds, result_list)


def inspect_results(ds, result_list):
    import pandas as pd
    import wbia.plottool as pt

    pd.set_option('display.max_rows', 20)
    pt.qt4ensure()

    isect_sets = [
        set(s1).intersection(set(s2))
        for s1, s2 in ut.combinations([result.df.index for result in result_list], 2)
    ]
    assert all([len(s) == 0 for s in isect_sets]), 'cv sets should not intersect'

    # Combine information from results
    df = pd.concat([result.df for result in result_list])
    df['hardness'] = 1 / df['easiness']
    df['aid'] = ut.take(ds.aids, df.index)
    df['target'] = ut.take(ds.target, df.index)
    df['failed'] = df['pred'] != df['target']

    report = sklearn.metrics.classification_report(
        y_true=df['target'], y_pred=df['pred'], target_names=result.ds.target_names
    )
    print(report)

    confusion = sklearn.metrics.confusion_matrix(df['target'], df['pred'])
    print('Confusion Matrix:')
    print(
        pd.DataFrame(
            confusion,
            columns=[m for m in result.ds.target_names],
            index=['gt ' + m for m in result.ds.target_names],
        )
    )

    def target_partition(target):
        df_chunk = df if target is None else df[df['target'] == target]
        df_chunk = df_chunk.take(df_chunk['hardness'].argsort())
        return df_chunk

    def grab_subchunk(frac, n, target):
        df_chunk = target_partition(target)
        sl = ut.snapped_slice(len(df_chunk), frac, n)
        print('sl = %r' % (sl,))
        idx = df_chunk.index[sl]
        df_chunk = df_chunk.loc[idx]
        min_frac = sl.start / len(df_chunk)
        max_frac = sl.stop / len(df_chunk)
        min_frac = sl.start
        max_frac = sl.stop
        place_name = 'hardness=%.2f (%d-%d)' % (frac, min_frac, max_frac)
        if target is not None:
            df_chunk.nice = place_name + ' ' + ds.target_names[target]
        else:
            df_chunk.nice = place_name
        return df_chunk

    def grab_subchunk2(df_chunk, frac, n):
        sl = ut.snapped_slice(len(df_chunk), frac, n)
        print('sl = %r' % (sl,))
        idx = df_chunk.index[sl]
        df_chunk = df_chunk.loc[idx]
        min_frac = sl.start / len(df_chunk)
        max_frac = sl.stop / len(df_chunk)
        min_frac = sl.start
        max_frac = sl.stop
        place_name = 'hardness=%.2f (%d-%d)' % (frac, min_frac, max_frac)
        if target is not None:
            df_chunk.nice = place_name + ' ' + ds.target_names[target]
        else:
            df_chunk.nice = place_name
        return df_chunk

    # Look at hardest train cases

    # Look at hardest test cases
    if True:
        # n = 4
        fracs = [0.0, 0.7, 0.8, 0.9, 1.0]
        view_targets = ds.target_labels
        n = 8 // len(view_targets)
    else:
        view_targets = [ut.listfind(ds.target_names.tolist(), 'healthy')]
        # fracs = [0.0, .7, .8, .9, 1.0]
        fracs = [0.45, 0.5, 0.55, 0.6, 0.62]
        fracs = [0.72, 0.82, 0.84, 0.88]
        fracs = [0.73, 0.83, 0.835, 0.89]
        fracs = [0.73, 0.83, 0.835, 0.89]
        fracs = [0.735, 0.833, 0.837, 0.934]
        fracs = [0.2, 0.65, 0.75, 0.85, 0.95]
        fracs = [0.3, 0.4, 0.67, 0.77, 0.87, 0.92]
        n = 8 // len(view_targets)

    if False:
        view_targets = [ut.listfind(ds.target_names.tolist(), 'healthy')]
        target_dfs = [target_partition(target) for target in view_targets]
        critical_points = [np.where(_df['failed'])[0][0] for _df in target_dfs]
        critical_fracs = [_pt / len(_df) for _pt, _df in zip(critical_points, target_dfs)]
        n = 8 * 5
        frac = critical_fracs[0]
        frac += 0.1
        _df = target_dfs[0]
        df_part = grab_subchunk2(_df, frac, n)
        df_chunks = [df_part.iloc[x] for x in ut.ichunks(range(len(df_part)), 8)]
    else:
        df_chunks = [
            grab_subchunk(frac, n, target) for frac in fracs for target in view_targets
        ]

    ibs = ds.ibs
    config = ds.config
    from wbia_cnn import draw_results

    inter = draw_results.make_InteractClasses(
        ibs, config, df_chunks, nCols=len(view_targets)
    )
    inter.start()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.scripts.classify_shark
        python -m wbia.scripts.classify_shark --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
