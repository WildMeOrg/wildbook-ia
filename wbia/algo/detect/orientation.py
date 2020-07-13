# -*- coding: utf-8 -*-
"""Interface to Lightnet object proposals."""
from __future__ import absolute_import, division, print_function
from os.path import expanduser, join
from wbia import constants as const
import utool as ut
import numpy as np
import cv2
import random
import tqdm
import time
import os
import copy
import PIL

(print, rrr, profile) = ut.inject2(__name__, '[orientation]')


PARALLEL = not const.CONTAINERIZED
INPUT_SIZE = 224


ARCHIVE_URL_DICT = {}


if not ut.get_argflag('--no-pytorch'):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torchvision

        print('PyTorch Version: ', torch.__version__)
        print('Torchvision Version: ', torchvision.__version__)
    except ImportError:
        print('WARNING Failed to import pytorch. ' 'PyTorch is unavailable')
        if ut.SUPER_STRICT:
            raise

    try:
        import imgaug  # NOQA

        class Augmentations(object):
            def __call__(self, img):
                img = np.array(img)
                return self.aug.augment_image(img)

        class TrainAugmentations(Augmentations):
            def __init__(self, blur=True, flip=False, rotate=10, shear=10, **kwargs):
                from imgaug import augmenters as iaa

                sequence = []

                sequence += [
                    iaa.Scale((INPUT_SIZE, INPUT_SIZE)),
                    iaa.ContrastNormalization((0.75, 1.25)),
                    iaa.AddElementwise((-10, 10), per_channel=0.5),
                    iaa.AddToHueAndSaturation(value=(-20, 20), per_channel=True),
                    iaa.Multiply((0.75, 1.25)),
                ]
                sequence += [
                    iaa.PiecewiseAffine(scale=(0.0005, 0.005)),
                    iaa.Affine(
                        rotate=(-rotate, rotate), shear=(-shear, shear), mode='symmetric',
                    ),
                    iaa.Grayscale(alpha=(0.0, 0.5)),
                ]
                if flip:
                    sequence += [
                        iaa.Fliplr(0.5),
                    ]
                if blur:
                    sequence += [
                        iaa.Sometimes(0.01, iaa.GaussianBlur(sigma=(0, 1.0))),
                    ]
                self.aug = iaa.Sequential(sequence)

        class ValidAugmentations(Augmentations):
            def __init__(self, **kwargs):
                from imgaug import augmenters as iaa

                self.aug = iaa.Sequential([iaa.Scale((INPUT_SIZE, INPUT_SIZE))])

        AUGMENTATION = {
            'train': TrainAugmentations,
            'val': ValidAugmentations,
            'test': ValidAugmentations,
        }
    except ImportError:
        AUGMENTATION = {}
        print(
            'WARNING Failed to import imgaug. '
            'install with pip install git+https://github.com/aleju/imgaug'
        )
        if ut.SUPER_STRICT:
            raise


def _init_transforms(**kwargs):
    TRANSFORMS = {
        phase: torchvision.transforms.Compose(
            [
                AUGMENTATION[phase](**kwargs),
                lambda array: PIL.Image.fromarray(array),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        for phase in AUGMENTATION.keys()
    }

    return TRANSFORMS


class ImageFilePathList(torch.utils.data.Dataset):
    def __init__(self, filepaths, targets=None, transform=None, target_transform=None):
        from torchvision.datasets.folder import default_loader

        self.targets = targets is not None

        args = (filepaths, targets,) if self.targets else (filepaths,)
        self.samples = list(zip(*args))

        if self.targets:
            self.classes = sorted(set(ut.take_column(self.targets, 1)))
            self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        else:
            self.classes, self.class_to_idx = None, None

        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.samples[index]

        if self.targets:
            path, target = sample
        else:
            path = sample[0]
            target = None

        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        result = (sample, target,) if self.targets else (sample,)

        return result

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        )
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        )
        return fmt_str


class StratifiedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, phase, multiplier=1.0):
        self.dataset = dataset
        self.phase = phase
        self.training = self.phase == 'train'

        self.labels = np.array(ut.take_column(dataset.samples, 1))
        self.classes = set(self.labels)

        self.indices = {
            cls: list(np.where(self.labels == cls)[0]) for cls in self.classes
        }
        self.counts = {cls: len(self.indices[cls]) for cls in self.classes}
        self.min = min(self.counts.values())
        self.min = int(np.around(multiplier * self.min))

        if self.training:
            self.total = 0
            for cls in self.indices:
                num_in_class = len(self.indices[cls])
                num_samples = min(self.min, num_in_class)
                self.total += num_samples
        else:
            self.total = len(self.labels)

        args = (
            self.phase,
            len(self.labels),
            len(self.classes),
            self.min,
            self.total,
            multiplier,
        )
        print(
            'Initialized Sampler for %r (sampling %d for %d classes | min %d per class, %d total, %0.02f multiplier)'
            % args
        )

    def __iter__(self):
        if self.training:
            ret_list = []
            for cls in self.indices:
                num_in_class = len(self.indices[cls])
                num_samples = min(self.min, num_in_class)
                ret_list += random.sample(self.indices[cls], num_samples)
            random.shuffle(ret_list)
        else:
            ret_list = range(self.total)
        assert len(ret_list) == self.total
        return iter(ret_list)

    def __len__(self):
        return self.total


def finetune(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=128):
    phases = ['train', 'val']

    start = time.time()

    best_accuracy = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    last_loss = {}
    best_loss = {}

    for epoch in range(num_epochs):
        start_batch = time.time()

        lr = optimizer.param_groups[0]['lr']
        print('Epoch {}/{} (lr = {:0.06f})'.format(epoch, num_epochs - 1, lr))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            seen = 0
            for inputs, labels in tqdm.tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                seen += len(inputs)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / seen
            epoch_acc = running_corrects.double() / seen

            last_loss[phase] = epoch_loss
            if phase not in best_loss:
                best_loss[phase] = np.inf

            flag = epoch_loss < best_loss[phase]
            if flag:
                best_loss[phase] = epoch_loss

            print(
                '{:<5} Loss: {:.4f} Acc: {:.4f} {}'.format(
                    phase, epoch_loss, epoch_acc, '!' if flag else ''
                )
            )

            # deep copy the model
            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                print('\tFound better model!')
                best_model_state = copy.deepcopy(model.state_dict())
            if phase == 'val':
                scheduler.step(epoch_loss)

                time_elapsed_batch = time.time() - start_batch
                print(
                    'time: {:.0f}m {:.0f}s'.format(
                        time_elapsed_batch // 60, time_elapsed_batch % 60
                    )
                )

                ratio = last_loss['train'] / last_loss['val']
                print('ratio: {:.04f}'.format(ratio))

        print('\n')

    time_elapsed = time.time() - start
    print(
        'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print('Best val Acc: {:4f}'.format(best_accuracy))

    # load best model weights
    model.load_state_dict(best_model_state)
    return model


def visualize_augmentations(dataset, augmentation, tag, num_per_class=10, **kwargs):
    import matplotlib.pyplot as plt

    samples = dataset.samples
    flags = np.array(ut.take_column(samples, 1))
    print('Dataset %r has %d samples' % (tag, len(flags),))

    indices = []
    for flag in set(flags):
        index_list = list(np.where(flags == flag)[0])
        random.shuffle(index_list)
        indices += index_list[:num_per_class]

    samples = ut.take(samples, indices)
    paths = ut.take_column(samples, 0)
    flags = ut.take_column(samples, 1)

    images = [np.array(cv2.imread(path)) for path in paths]
    images = [image[:, :, ::-1] for image in images]

    images_ = []
    for image, flag in zip(images, flags):
        image_ = image.copy()
        color = (0, 255, 0) if flag else (255, 0, 0)
        cv2.rectangle(image_, (1, 1), (INPUT_SIZE - 1, INPUT_SIZE - 1), color, 3)
        images_.append(image_)
    canvas = np.hstack(images_)
    canvas_list = [canvas]

    augment = augmentation(**kwargs)
    for index in range(len(indices) - 1):
        print(index)
        images_ = [augment(image.copy()) for image in images]
        canvas = np.hstack(images_)
        canvas_list.append(canvas)
    canvas = np.vstack(canvas_list)

    canvas_filepath = expanduser(
        join('~', 'Desktop', 'orientation-augmentation-%s.png' % (tag,))
    )
    plt.imsave(canvas_filepath, canvas)


def train(
    data_path,
    output_path,
    batch_size=48,
    class_weights={},
    multi=PARALLEL,
    sample_multiplier=1.0,
    **kwargs,
):
    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    using_gpu = str(device) != 'cpu'

    phases = ['train', 'val']

    print('Initializing Datasets and Dataloaders...')

    # Create training and validation datasets
    transforms = _init_transforms(**kwargs)
    datasets = {
        phase: torchvision.datasets.ImageFolder(
            os.path.join(data_path, phase), transforms[phase]
        )
        for phase in phases
    }

    # Create training and validation dataloaders
    dataloaders = {
        phase: torch.utils.data.DataLoader(
            datasets[phase],
            sampler=StratifiedSampler(
                datasets[phase], phase, multiplier=sample_multiplier
            ),
            batch_size=batch_size,
            num_workers=batch_size // 8,
            pin_memory=using_gpu,
        )
        for phase in phases
    }

    train_classes = datasets['train'].classes
    val_classes = datasets['val'].classes

    assert len(train_classes) == len(val_classes)
    num_classes = len(train_classes)

    print('Initializing Model...')

    # Initialize the model for this run
    model = torchvision.models.densenet201(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # Send the model to GPU
    model = model.to(device)

    # Multi-GPU
    if multi:
        print('USING MULTI-GPU MODEL')
        model = nn.DataParallel(model)

    print('Print Examples of Training Augmentation...')

    for phase in phases:
        visualize_augmentations(datasets[phase], AUGMENTATION[phase], phase, **kwargs)

    print('Initializing Optimizer...')

    # print('Params to learn:')
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            # print('\t', name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=10, min_lr=1e-6
    )

    # Get weights for the class
    class_index_list = list(dataloaders['train'].dataset.class_to_idx.items())
    index_class_list = [class_index[::-1] for class_index in class_index_list]
    weight = torch.tensor(
        [class_weights.get(class_, 1.0) for index, class_ in sorted(index_class_list)]
    )
    weight = weight.to(device)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss(weight=weight)

    print('Start Training...')

    # Train and evaluate
    model = finetune(model, dataloaders, criterion, optimizer, scheduler, device)

    ut.ensuredir(output_path)
    weights_path = os.path.join(output_path, 'classifier.orientation.weights')
    weights = {
        'state': copy.deepcopy(model.state_dict()),
        'classes': train_classes,
    }
    torch.save(weights, weights_path)

    return weights_path


def test_single(filepath_list, weights_path, batch_size=1792, multi=PARALLEL, **kwargs):

    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    using_gpu = str(device) != 'cpu'

    print('Initializing Datasets and Dataloaders...')

    # Create training and validation datasets
    transforms = _init_transforms(**kwargs)
    dataset = ImageFilePathList(filepath_list, transform=transforms['test'])

    # Create training and validation dataloaders
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=using_gpu
    )

    print('Initializing Model...')
    try:
        weights = torch.load(weights_path)
    except RuntimeError:
        weights = torch.load(weights_path, map_location='cpu')
    state = weights['state']
    classes = weights['classes']

    num_classes = len(classes)

    # Initialize the model for this run
    model = torchvision.models.densenet201()
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # Convert any weights to non-parallel version
    from collections import OrderedDict

    new_state = OrderedDict()
    for k, v in state.items():
        k = k.replace('module.', '')
        new_state[k] = v

    # Load state without parallel
    model.load_state_dict(new_state)

    # Add softmax
    model.classifier = nn.Sequential(model.classifier, nn.LogSoftmax(), nn.Softmax())

    # Make parallel at end
    if multi:
        print('USING MULTI-GPU MODEL')
        model = nn.DataParallel(model)

    # Send the model to GPU
    model = model.to(device)

    model.eval()

    start = time.time()

    counter = 0
    outputs = []
    for (inputs,) in tqdm.tqdm(dataloader, desc='test'):
        print('Loading batch %d from disk' % (counter,))
        inputs = inputs.to(device)
        print('Moving batch %d to GPU' % (counter,))
        with torch.set_grad_enabled(False):
            print('Pre-model inference %d' % (counter,))
            output = model(inputs)
            print('Post-model inference %d' % (counter,))
            outputs += output.tolist()
            print('Outputs done %d' % (counter,))
        counter += 1

    time_elapsed = time.time() - start
    print(
        'Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    result_list = []
    for output in outputs:
        result = dict(zip(classes, output))
        result_list.append(result)

    return result_list


def test_ensemble(
    filepath_list,
    weights_path_list,
    classifier_weight_filepath,
    ensemble_index,
    ibs=None,
    gid_list=None,
    multiclass=False,
    **kwargs,
):

    if ensemble_index is not None:
        assert 0 <= ensemble_index and ensemble_index < len(weights_path_list)
        weights_path_list = [weights_path_list[ensemble_index]]
        assert len(weights_path_list) > 0

    cached = False
    try:
        assert ensemble_index is None, 'Do not use depc on individual model computation'
        assert None not in [ibs, gid_list], 'Needs to have access to depc'
        assert len(filepath_list) == len(gid_list)

        results_list = []
        for model_index in range(len(weights_path_list)):
            if multiclass:
                classifier_two_weight_filepath_ = '%s:%d' % (
                    classifier_weight_filepath,
                    model_index,
                )
                config = {
                    'classifier_two_algo': 'orientation',
                    'classifier_two_weight_filepath': classifier_two_weight_filepath_,
                }
                scores_list = ibs.depc_image.get_property(
                    'classifier_two', gid_list, 'scores', config=config
                )
                result_list = []
                for score_dict in scores_list:
                    result = score_dict
                    result_list.append(result)
            else:
                classifier_weight_filepath_ = '%s:%d' % (
                    classifier_weight_filepath,
                    model_index,
                )
                config = {
                    'classifier_algo': 'orientation',
                    'classifier_weight_filepath': classifier_weight_filepath_,
                }
                prediction_list = ibs.depc_image.get_property(
                    'classifier', gid_list, 'class', config=config
                )
                confidence_list = ibs.depc_image.get_property(
                    'classifier', gid_list, 'score', config=config
                )
                result_list = []
                for prediction, confidence in zip(prediction_list, confidence_list):
                    # DO NOT REMOVE THIS ASSERT
                    assert prediction in set(
                        ['negative', 'positive']
                    ), 'Cannot use this method, need to implement classifier_two in depc'
                    if prediction == 'positive':
                        pscore = confidence
                        nscore = 1.0 - pscore
                    else:
                        nscore = confidence
                        pscore = 1.0 - nscore
                    result = {
                        'positive': pscore,
                        'negative': nscore,
                    }
                    result_list.append(result)
            assert len(result_list) == len(gid_list)
            results_list.append(result_list)
        assert len(results_list) == len(weights_path_list)
        cached = True
    except AssertionError:
        cached = False

    if not cached:
        # Use local implementation, due to error or not valid config
        results_list = []
        for weights_path in weights_path_list:
            result_list = test_single(filepath_list, weights_path)
            results_list.append(result_list)

    for result_list in zip(*results_list):
        merged = {}
        for result in result_list:
            for key in result:
                if key not in merged:
                    merged[key] = []
                merged[key].append(result[key])
        for key in merged:
            value_list = merged[key]
            merged[key] = sum(value_list) / len(value_list)

        yield merged


def test(
    gpath_list,
    classifier_weight_filepath=None,
    return_dict=False,
    multiclass=False,
    **kwargs,
):
    from wbia.detecttools.directory import Directory

    # Get correct weight if specified with shorthand
    archive_url = None

    ensemble_index = None
    if classifier_weight_filepath is not None and ':' in classifier_weight_filepath:
        assert classifier_weight_filepath.count(':') == 1
        classifier_weight_filepath, ensemble_index = classifier_weight_filepath.split(':')
        ensemble_index = int(ensemble_index)

    if classifier_weight_filepath in ARCHIVE_URL_DICT:
        archive_url = ARCHIVE_URL_DICT[classifier_weight_filepath]
        archive_path = ut.grab_file_url(archive_url, appname='wbia', check_hash=True)
    else:
        print(
            'classifier_weight_filepath %r not recognized' % (classifier_weight_filepath,)
        )
        raise RuntimeError

    assert os.path.exists(archive_path)
    archive_path = ut.truepath(archive_path)

    ensemble_path = archive_path.strip('.zip')
    if not os.path.exists(ensemble_path):
        ut.unarchive_file(archive_path, output_dir=ensemble_path)

    assert os.path.exists(ensemble_path)
    direct = Directory(ensemble_path, include_file_extensions=['weights'], recursive=True)
    weights_path_list = direct.files()
    weights_path_list = sorted(weights_path_list)
    assert len(weights_path_list) > 0

    kwargs.pop('classifier_algo', None)

    print(
        'Using weights in the ensemble, index %r: %s '
        % (ensemble_index, ut.repr3(weights_path_list),)
    )
    result_list = test_ensemble(
        gpath_list,
        weights_path_list,
        classifier_weight_filepath,
        ensemble_index,
        multiclass=multiclass,
        **kwargs,
    )
    for result in result_list:
        best_key = None
        best_score = -1.0
        for key, score in result.items():
            if score > best_score:
                best_key = key
                best_score = score
        assert best_score >= 0.0 and best_key is not None
        if return_dict:
            yield best_score, best_key, result
        else:
            yield best_score, best_key


def test_dict(gpath_list, classifier_weight_filepath=None, return_dict=None, **kwargs):
    result_gen = test(
        gpath_list,
        classifier_weight_filepath=classifier_weight_filepath,
        return_dict=True,
        **kwargs,
    )

    for result in result_gen:
        best_score, best_key, result_dict = result
        best_key = best_key.split(':')
        if len(best_key) == 1:
            best_species = best_key
            best_viewpoint = None
        elif len(best_key) == 2:
            best_species, best_viewpoint = best_key
        else:
            raise ValueError('Invalid key %r' % (best_key,))

        yield (
            best_score,
            best_species,
            best_viewpoint,
            'UNKNOWN',
            0.0,
            result_dict,
        )


def features(filepath_list, batch_size=512, multi=PARALLEL, **kwargs):
    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    using_gpu = str(device) != 'cpu'

    print('Initializing Datasets and Dataloaders...')

    # Create training and validation datasets
    transforms = _init_transforms(**kwargs)
    dataset = ImageFilePathList(filepath_list, transform=transforms['test'])

    # Create training and validation dataloaders
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=using_gpu
    )

    # Initialize the model for this run
    model = torchvision.models.densenet201(pretrained=True)

    # Send the model to GPU
    model = model.to(device)

    if multi:
        print('USING MULTI-GPU MODEL')
        model = nn.DataParallel(model)

    model.eval()

    start = time.time()

    outputs = []
    for (inputs,) in tqdm.tqdm(dataloader, desc='test'):
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            output = model(inputs)
            outputs += output.tolist()

    outputs = np.array(outputs, dtype=np.float32)
    time_elapsed = time.time() - start
    print(
        'Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return outputs
