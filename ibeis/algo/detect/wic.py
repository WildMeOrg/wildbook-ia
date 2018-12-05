# -*- coding: utf-8 -*-
"""Interface to Lightnet object proposals."""
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
import cv2
import random
import tqdm
import time
import os
import copy
import PIL
(print, rrr, profile) = ut.inject2(__name__, '[wic]')


INPUT_SIZE = 224


ARCHIVE_URL_DICT = {
    'vulcan': 'https://cthulhu.dyn.wildme.io/public/models/classifier2.vulcan.tar',
    None:     'https://cthulhu.dyn.wildme.io/public/models/classifier2.vulcan.tar',
}


if not ut.get_argflag('--no-pytorch'):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torchvision
        print('PyTorch Version: ', torch.__version__)
        print('Torchvision Version: ', torchvision.__version__)
    except ImportError:
        print('WARNING Failed to import pytorch. '
              'PyTorch is unavailable')
        if ut.SUPER_STRICT:
            raise

    try:
        import imgaug  # NOQA

        class Augmentations(object):
            def __call__(self, img):
                img = np.array(img)
                return self.aug.augment_image(img)

        class TrainAugmentations(Augmentations):
            def __init__(self):
                from imgaug import augmenters as iaa
                self.aug = iaa.Sequential([
                    iaa.Scale((INPUT_SIZE, INPUT_SIZE)),
                    iaa.AddElementwise((-20, 20), per_channel=0.5),
                    iaa.AddToHueAndSaturation(value=(-20, 20), per_channel=True),
                    iaa.Grayscale(alpha=(0.0, 0.5)),
                    iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 2.0))),
                    iaa.Affine(rotate=(-30, 30), shear=(-14, 14), mode='symmetric'),
                    iaa.Fliplr(0.5),
                ])

        class ValidAugmentations(Augmentations):
            def __init__(self):
                from imgaug import augmenters as iaa
                self.aug = iaa.Sequential([
                    iaa.Scale((INPUT_SIZE, INPUT_SIZE)),
                ])

        AGUEMTNATION = {
            'train': TrainAugmentations,
            'val':   ValidAugmentations,
            'test':  ValidAugmentations,
        }

        TRANSFORMS = {
            phase: torchvision.transforms.Compose([
                AGUEMTNATION[phase](),
                lambda array: PIL.Image.fromarray(array),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            for phase in AGUEMTNATION.keys()
        }
    except ImportError:
        AGUEMTNATION = {}
        TRANSFORMS = {}
        print('WARNING Failed to import imgaug. '
              'install with pip install git+https://github.com/aleju/imgaug')
        if ut.SUPER_STRICT:
            raise


class ImageFilePathList(torch.utils.data.Dataset):

    def __init__(self, filepaths, targets=None, transform=None, target_transform=None):
        from torchvision.datasets.folder import default_loader
        args = (filepaths, ) if targets is None else (filepaths, targets, )
        self.samples = list(zip(*args))

        if targets is None:
            self.classes, self.class_to_idx = None, None
        else:
            self.classes = sorted(set(targets))
            self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = filepaths
        self.targets = targets

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
        path = self.samples[index]
        sample = self.loader(path)
        if self.targets is None:
            target = None
        if self.transform is not None:
            sample = self.transform(sample)
        if target is not None and self.target_transform is not None:
            target = self.target_transform(target)

        result = (sample, ) if target is None else (sample, target, )
        return result

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class StratifiedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, phase):
        self.dataset = dataset
        self.phase = phase
        self.training = self.phase == 'train'

        self.labels = np.array(ut.take_column(dataset.samples, 1))
        self.classes = set(self.labels)

        self.indices = {
            cls: list(np.where(self.labels == cls)[0])
            for cls in self.classes
        }
        self.counts = {
            cls: len(self.indices[cls])
            for cls in self.classes
        }
        self.min = min(self.counts.values())

        if self.training:
            self.total = self.min * len(self.classes)
        else:
            self.total = len(self.labels)

        args = (self.phase, len(self.labels), len(self.classes), self.min, self.total, )
        print('Initialized Sampler for %r (sampling %d for %d classes | min %d per class, %d total)' % args)

    def __iter__(self):
        if self.training:
            ret_list = []
            for cls in self.indices:
                ret_list += random.sample(self.indices[cls], self.min)
            random.shuffle(ret_list)
        else:
            ret_list = range(self.total)
        assert len(ret_list) == self.total
        return iter(ret_list)

    def __len__(self):
        return self.total


def finetune(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=64):
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
                model.eval()   # Set model to evaluate mode

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

            print('{:<5} Loss: {:.4f} Acc: {:.4f} {}'.format(phase, epoch_loss, epoch_acc, '!' if flag else ''))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_state = copy.deepcopy(model.state_dict())
            if phase == 'val':
                scheduler.step(epoch_loss)

                time_elapsed_batch = time.time() - start_batch
                print('time: {:.0f}m {:.0f}s'.format(time_elapsed_batch // 60, time_elapsed_batch % 60))

                ratio = last_loss['train'] / last_loss['val']
                print('ratio: {:.04f}'.format(ratio))

        print('\n')

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_accuracy))

    # load best model weights
    model.load_state_dict(best_model_state)
    return model


def visualize_augmentations(dataset, augmentation, tag, num_per_class=5):
    import matplotlib.pyplot as plt
    samples = dataset.samples
    flags = np.array(ut.take_column(samples, 1))
    print('Dataset %r has %d samples' % (tag, len(flags), ))

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

    augment = augmentation()
    for index in range(len(indices) - 1):
        print(index)
        images_ = [augment(image.copy()) for image in images]
        canvas = np.hstack(images_)
        canvas_list.append(canvas)
    canvas = np.vstack(canvas_list)

    canvas_filepath = '/home/jason.parham/Desktop/augmentation-%s.png' % (tag, )
    plt.imsave(canvas_filepath, canvas)


def train(data_path, output_path, batch_size=32):

    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    using_gpu = device != 'cpu'

    phases = ['train', 'val']

    print('Initializing Datasets and Dataloaders...')

    # Create training and validation datasets
    datasets = {
        phase: torchvision.datasets.ImageFolder(os.path.join(data_path, phase), TRANSFORMS[phase])
        for phase in phases
    }

    # Create training and validation dataloaders
    dataloaders = {
        phase: torch.utils.data.DataLoader(
            datasets[phase],
            sampler=StratifiedSampler(datasets[phase], phase),
            batch_size=batch_size,
            num_workers=batch_size // 8,
            pin_memory=using_gpu
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

    print('Print Examples of Training Augmentation...')

    for phase in phases:
        visualize_augmentations(datasets[phase], AGUEMTNATION[phase], phase)

    print('Initializing Optimizer...')

    # print('Params to learn:')
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            # print('\t', name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, min_lr=1e-5)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    print('Start Training...')

    # Train and evaluate
    model = finetune(model, dataloaders, criterion, optimizer, scheduler, device)

    ut.ensuredir(output_path)
    weights_path = os.path.join(output_path, 'classifier.weights')
    weights = {
        'state':   copy.deepcopy(model.state_dict()),
        'classes': train_classes,
    }
    torch.save(weights, weights_path)

    return weights_path


def test_single(filepath_list, weights_path, batch_size=512):

    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    using_gpu = device != 'cpu'

    print('Initializing Datasets and Dataloaders...')

    # Create training and validation datasets
    dataset = ImageFilePathList(filepath_list, transform=TRANSFORMS['test'])

    # Create training and validation dataloaders
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=StratifiedSampler(dataset, 'test'),
        batch_size=batch_size,
        num_workers=batch_size // 8,
        pin_memory=using_gpu
    )

    print('Initializing Model...')
    weights = torch.load(weights_path)
    state   = weights['state']
    classes = weights['classes']

    num_classes = len(classes)

    # Initialize the model for this run
    model = torchvision.models.densenet201()
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    model.load_state_dict(state)

    # Add LogSoftmax and Softmax to network output
    model.classifier = nn.Sequential(
        model.classifier,
        nn.LogSoftmax(),
        nn.Softmax()
    )

    # Send the model to GPU
    model = model.to(device)
    model.eval()

    start = time.time()

    outputs = []
    for inputs, in tqdm.tqdm(dataloader, desc='test'):
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            output = model(inputs)
            outputs += output.tolist()

    time_elapsed = time.time() - start
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    result_list = []
    for output in outputs:
        result = dict(zip(classes, output))
        result_list.append(result)

    return result_list


def test_ensemble(filepath_list, weights_path_list, **kwargs):

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


def test(gpath_list, classifier_weight_filepath=None, **kwargs):
    # Get correct weight if specified with shorthand
    archive_url = None

    ensemble_index = None
    if classifier_weight_filepath is not None and ':' in classifier_weight_filepath:
        assert classifier_weight_filepath.count(':') == 1
        classifier_weight_filepath, ensemble_index = classifier_weight_filepath.split(':')
        ensemble_index = int(ensemble_index)

    if classifier_weight_filepath in ARCHIVE_URL_DICT:
        archive_url = ARCHIVE_URL_DICT[classifier_weight_filepath]
        archive_path = ut.grab_file_url(archive_url, appname='vulcan', check_hash=True)
    else:
        print('classifier_weight_filepath %r not recognized' % (classifier_weight_filepath, ))
        raise RuntimeError

    kwargs.pop('classifier_algo', None)

    assert os.path.exists(archive_path)
    archive_path = ut.truepath(archive_path)

    ensemble_path = ut.unarchive_file(archive_path)
    ensemble_path = os.path.join(ensemble_path, 'ensemble', '*.weights')
    weights_path_list = ut.glob(ensemble_path)

    weights_path_list = sorted(weights_path_list)

    if ensemble_index is not None:
        assert 0 <= ensemble_index and ensemble_index < len(weights_path_list)
        weights_path_list = [ weights_path_list[ensemble_index] ]

    print('Using weights in the ensemble: %s ' % (ut.repr3(weights_path_list), ))
    result_list = test_ensemble(gpath_list, weights_path_list, **kwargs)
    for result in result_list:
        if result['positive'] > 0.5:
            cls = 'positive'
        else:
            cls = 'negative'
        score = result[cls]

        yield score, cls
