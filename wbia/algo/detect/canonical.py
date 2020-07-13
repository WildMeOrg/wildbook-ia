# -*- coding: utf-8 -*-
"""Interface to Lightnet object proposals."""
from __future__ import absolute_import, division, print_function
from os.path import expanduser, join
import utool as ut
import numpy as np
import cv2
import random
import tqdm
import time
import os
import copy
import PIL

(print, rrr, profile) = ut.inject2(__name__, '[canonical]')


INPUT_SIZE = 224


ARCHIVE_URL_DICT = {
    'canonical_zebra_grevys_v1': 'https://wildbookiarepository.azureedge.net/models/localizer.canonical.zebra_grevys.v1.zip',
    'canonical_zebra_grevys_v2': 'https://wildbookiarepository.azureedge.net/models/localizer.canonical.zebra_grevys.v2.zip',
    'canonical_zebra_grevys_v3': 'https://wildbookiarepository.azureedge.net/models/localizer.canonical.zebra_grevys.v3.zip',
    'canonical_zebra_grevys_v4': 'https://wildbookiarepository.azureedge.net/models/localizer.canonical.zebra_grevys.v4.zip',
    'canonical_zebra_grevys_v5': 'https://wildbookiarepository.azureedge.net/models/localizer.canonical.zebra_grevys.v5.zip',
    'canonical_zebra_grevys_v6': 'https://wildbookiarepository.azureedge.net/models/localizer.canonical.zebra_grevys.v6.zip',
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
            def __init__(self):
                from imgaug import augmenters as iaa

                self.aug = iaa.Sequential(
                    [
                        iaa.Scale((INPUT_SIZE, INPUT_SIZE)),
                        iaa.ContrastNormalization((0.75, 1.25)),
                        iaa.AddElementwise((-20, 20), per_channel=0.5),
                        iaa.AddToHueAndSaturation(value=(-5, 5), per_channel=True),
                        iaa.Multiply((0.75, 1.25)),
                        # iaa.Dropout(p=(0.0, 0.1)),
                        iaa.PiecewiseAffine(scale=(0.0001, 0.0005)),
                        iaa.Affine(rotate=(-1, 1), shear=(-1, 1), mode='symmetric'),
                        iaa.Grayscale(alpha=(0.0, 0.25)),
                    ]
                )

        class ValidAugmentations(Augmentations):
            def __init__(self):
                from imgaug import augmenters as iaa

                self.aug = iaa.Sequential([iaa.Scale((INPUT_SIZE, INPUT_SIZE))])

        AGUEMTNATION = {
            'train': TrainAugmentations,
            'val': ValidAugmentations,
            'test': ValidAugmentations,
        }

        TRANSFORMS = {
            phase: torchvision.transforms.Compose(
                [
                    AGUEMTNATION[phase](),
                    lambda array: PIL.Image.fromarray(array),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
            for phase in AGUEMTNATION.keys()
        }
    except ImportError:
        AGUEMTNATION = {}
        TRANSFORMS = {}
        print(
            'WARNING Failed to import imgaug. '
            'install with pip install git+https://github.com/aleju/imgaug'
        )
        if ut.SUPER_STRICT:
            raise


class ImageFilePathList(torch.utils.data.Dataset):
    def __init__(self, filepaths, targets=True, transform=None, target_transform=None):
        from torchvision.datasets.folder import default_loader

        self.targets = targets

        if self.targets:
            targets = []
            for filepath in filepaths:
                path, ext = os.path.splitext(filepath)
                target = '%s.csv' % (path,)
                assert os.path.exists(target), 'Missing target %s for %s' % (
                    target,
                    filepath,
                )
                targets.append(target)
            args = (
                filepaths,
                targets,
            )
        else:
            args = (filepaths,)

        self.samples = list(zip(*args))

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
            with open(target, 'r') as target_file:
                target_str = target_file.readline().strip().split(',')
            assert len(target_str) == 4
            target = list(map(float, target_str))
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


def finetune(
    model, dataloaders, optimizer, scheduler, device, num_epochs=128, under=1.0, over=1.0,
):
    phases = ['train', 'val']

    start = time.time()

    best_model_state = copy.deepcopy(model.state_dict())

    last_loss = {}
    best_loss = {}
    best_correction = None

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

            running_loss_ = np.zeros((1, 4))
            running_loss_under_ = np.zeros((1, 4))
            running_loss_over_ = np.zeros((1, 4))
            running_loss = 0.0

            # Iterate over data.
            seen = 0
            for inputs, labels in tqdm.tqdm(dataloaders[phase], desc=phase):

                labels = torch.tensor(list(zip(*labels)), dtype=torch.float32)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)

                    undershoots = labels - outputs
                    overshoots = outputs - labels

                    # partition
                    undershoots[undershoots < 0] = 0
                    overshoots[overshoots < 0] = 0

                    # Square
                    undershoots = undershoots * undershoots
                    overshoots = overshoots * overshoots

                    # Weighted
                    undershoots *= under
                    overshoots *= over

                    # Sum
                    error = undershoots + overshoots

                    # error = outputs - labels
                    # error = error * error

                    # Bias towards bad instances
                    # loss_sorted, loss_index = torch.sort(loss_)
                    # loss_index += 1
                    # loss_index = torch.tensor(loss_index, dtype=loss_.dtype)
                    # loss_index = loss_index.to(device)
                    # loss_weighted = loss_ * loss_index
                    # loss = torch.sum(loss_weighted)

                    loss_ = torch.mean(error, 0)
                    loss_under_ = torch.mean(undershoots, 0)
                    loss_over_ = torch.mean(overshoots, 0)

                    loss = torch.sum(loss_)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                seen += len(inputs)
                running_loss += loss.item() * inputs.size(0)
                running_loss_ += np.array(loss_.tolist()) * inputs.size(0)
                running_loss_under_ += np.array(loss_under_.tolist()) * inputs.size(0)
                running_loss_over_ += np.array(loss_over_.tolist()) * inputs.size(0)

            epoch_loss = running_loss / seen
            epoch_loss_ = running_loss_[0] / seen
            epoch_loss_under_ = running_loss_under_[0] / seen
            epoch_loss_over_ = running_loss_over_[0] / seen

            last_loss[phase] = epoch_loss

            if phase not in best_loss:
                best_loss[phase] = np.inf

            best = epoch_loss < best_loss[phase]
            if best:
                best_loss[phase] = epoch_loss

            x0, y0, x1, y1 = epoch_loss_
            x0 *= INPUT_SIZE
            y0 *= INPUT_SIZE
            x1 *= INPUT_SIZE
            y1 *= INPUT_SIZE
            best_str = '!' if best else ''
            print(
                '{:<5} Loss: {:.4f}\t(X0: {:.1f}px Y0: {:.1f}px X1: {:.1f}px Y1: {:.1f}px)\t{}'.format(
                    phase, epoch_loss, x0, y0, x1, y1, best_str
                )
            )

            x0_, y0_, x1_, y1_ = epoch_loss_under_
            x0_ *= INPUT_SIZE
            y0_ *= INPUT_SIZE
            x1_ *= INPUT_SIZE
            y1_ *= INPUT_SIZE
            print(
                '{:<5} Under Loss: \t(X0: {:.1f}px Y0: {:.1f}px X1: {:.1f}px Y1: {:.1f}px)'.format(
                    phase, x0_, y0_, x1_, y1_
                )
            )

            x0_, y0_, x1_, y1_ = epoch_loss_over_
            x0_ *= INPUT_SIZE
            y0_ *= INPUT_SIZE
            x1_ *= INPUT_SIZE
            y1_ *= INPUT_SIZE
            print(
                '{:<5}  Over Loss: \t(X0: {:.1f}px Y0: {:.1f}px X1: {:.1f}px Y1: {:.1f}px)'.format(
                    phase, x0_, y0_, x1_, y1_
                )
            )

            if phase == 'val':
                if best:
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_correction = (
                        x0,
                        y0,
                        x1,
                        y1,
                    )

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
    print('Suggested correction offsets: %r' % (best_correction,))

    # load best model weights
    model.load_state_dict(best_model_state)
    return model


def visualize_augmentations(dataset, augmentation, tag, num=20):
    import matplotlib.pyplot as plt

    samples = dataset.samples
    print('Dataset %r has %d samples' % (tag, len(samples),))

    index_list = list(range(len(samples)))
    random.shuffle(index_list)
    indices = index_list[:num]

    samples = ut.take(samples, indices)
    image_paths = ut.take_column(samples, 0)
    bbox_paths = ut.take_column(samples, 1)

    images = [np.array(cv2.imread(image_path)) for image_path in image_paths]
    images = [image[:, :, ::-1] for image in images]

    images_ = []
    for image, bbox_path in zip(images, bbox_paths):
        with open(bbox_path, 'r') as bbox_file:
            bbox_str = bbox_file.readline().strip().split(',')

        assert len(bbox_str) == 4
        bbox = list(map(float, bbox_str))
        x0, y0, x1, y1 = bbox

        x0 = int(np.around(x0 * INPUT_SIZE))
        y0 = int(np.around(y0 * INPUT_SIZE))
        x1 = int(np.around(x1 * INPUT_SIZE))
        y1 = int(np.around(y1 * INPUT_SIZE))

        image_ = image.copy()
        color = (0, 255, 0)
        cv2.rectangle(image_, (x0, y0), (INPUT_SIZE - x1, INPUT_SIZE - y1), color, 3)
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

    canvas_filepath = expanduser(
        join('~', 'Desktop', 'canonical-augmentation-%s.png' % (tag,))
    )
    plt.imsave(canvas_filepath, canvas)


def train(data_path, output_path, batch_size=32):
    # Detect if we have a GPU available

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    using_gpu = str(device) != 'cpu'

    phases = ['train', 'val']

    print('Initializing Datasets and Dataloaders...')

    # Create training and validation datasets
    filepaths = {
        phase: ut.glob(os.path.join(data_path, phase, '*.png')) for phase in phases
    }

    datasets = {
        phase: ImageFilePathList(filepaths[phase], transform=TRANSFORMS[phase])
        for phase in phases
    }

    # Create training and validation dataloaders
    dataloaders = {
        phase: torch.utils.data.DataLoader(
            datasets[phase],
            batch_size=batch_size,
            num_workers=batch_size // 8,
            pin_memory=using_gpu,
        )
        for phase in phases
    }

    print('Initializing Model...')

    # Initialize the model for this run
    model = torchvision.models.densenet201(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 4),)

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
    optimizer = optim.SGD(params_to_update, lr=0.0005, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=16, min_lr=1e-6
    )

    print('Start Training...')

    # Train and evaluate
    model = finetune(model, dataloaders, optimizer, scheduler, device)

    ut.ensuredir(output_path)
    weights_path = os.path.join(output_path, 'localizer.canonical.weights')
    weights = {
        'state': copy.deepcopy(model.state_dict()),
    }
    torch.save(weights, weights_path)

    return weights_path


def test_single(filepath_list, weights_path, batch_size=512):

    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    using_gpu = str(device) != 'cpu'

    print('Initializing Datasets and Dataloaders...')

    # Create training and validation datasets
    dataset = ImageFilePathList(
        filepath_list, transform=TRANSFORMS['test'], targets=False
    )

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

    # Initialize the model for this run
    model = torchvision.models.densenet201()
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 4),)

    model.load_state_dict(state)

    # Add LogSoftmax and Softmax to network output

    # Send the model to GPU
    model = model.to(device)
    model.eval()

    start = time.time()

    outputs = []
    for (inputs,) in tqdm.tqdm(dataloader, desc='test'):
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            output = model(inputs)
            output = output.cpu()
            outputs.append(np.array(output))

    outputs = np.vstack(outputs)

    time_elapsed = time.time() - start
    print(
        'Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    classes = ['x0', 'y0', 'x1', 'y1']
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


def test(gpath_list, canonical_weight_filepath=None, **kwargs):
    from wbia.detecttools.directory import Directory

    # Get correct weight if specified with shorthand
    archive_url = None

    ensemble_index = None
    if canonical_weight_filepath is not None and ':' in canonical_weight_filepath:
        assert canonical_weight_filepath.count(':') == 1
        canonical_weight_filepath, ensemble_index = canonical_weight_filepath.split(':')
        ensemble_index = int(ensemble_index)

    if canonical_weight_filepath in ARCHIVE_URL_DICT:
        archive_url = ARCHIVE_URL_DICT[canonical_weight_filepath]
        archive_path = ut.grab_file_url(archive_url, appname='wbia', check_hash=True)
    else:
        raise RuntimeError(
            'canonical_weight_filepath %r not recognized' % (canonical_weight_filepath,)
        )

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

    if ensemble_index is not None:
        assert 0 <= ensemble_index and ensemble_index < len(weights_path_list)
        weights_path_list = [weights_path_list[ensemble_index]]
        assert len(weights_path_list) > 0

    print('Using weights in the ensemble: %s ' % (ut.repr3(weights_path_list),))
    result_list = test_ensemble(gpath_list, weights_path_list, **kwargs)
    for result in result_list:
        x0 = max(result['x0'], 0.0)
        y0 = max(result['y0'], 0.0)
        x1 = max(result['x1'], 0.0)
        y1 = max(result['y1'], 0.0)
        yield (
            x0,
            y0,
            x1,
            y1,
        )
