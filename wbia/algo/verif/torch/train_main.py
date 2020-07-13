# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join  # NOQA
import cv2
import numpy as np
import torch
import torch.nn
import utool as ut
import torchvision

print, rrr, profile = ut.inject2(__name__)


class LRSchedule(object):
    @staticmethod
    def exp(optimizer, epoch, init_lr=0.001, lr_decay_epoch=2):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr
        # epoch += 1
        if epoch % lr_decay_epoch == 0 and epoch != 0:
            lr *= 0.1

        if epoch % lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer, lr


def siam_vsone_train():
    r"""
    CommandLine:
        python -m wbia.algo.verif.torch.train_main siam_vsone_train

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.verif.torch.train_main import *  # NOQA
        >>> siam_vsone_train()
    """
    # wrapper around the RF vsone problem
    from wbia.algo.verif import vsone

    # pblm = vsone.OneVsOneProblem.from_empty('PZ_MTEST')
    pblm = vsone.OneVsOneProblem.from_empty('GZ_Master1')
    ibs = pblm.infr.ibs
    pblm.load_samples()
    samples = pblm.samples
    samples.print_info()
    xval_kw = pblm.xval_kw.asdict()
    skf_list = pblm.samples.stratified_kfold_indices(**xval_kw)

    def load_dataset(subset_idx):
        aids1, aids2 = pblm.samples.aid_pairs[subset_idx].T
        labels = pblm.samples['match_state'].y_enc[subset_idx]

        # train only on positive-vs-negative (ignore incomparable)
        labels = (labels == 1).astype(np.int64)

        chip_config = {'resize_dim': 'wh', 'dim_size': (224, 224)}
        img1_fpaths = ibs.depc_annot.get(
            'chips', aids1, read_extern=False, colnames='img', config=chip_config
        )
        img2_fpaths = ibs.depc_annot.get(
            'chips', aids2, read_extern=False, colnames='img', config=chip_config
        )
        dataset = LabeledPairDataset(img1_fpaths, img2_fpaths, labels)
        return dataset

    learn_idx, test_idx = skf_list[0]
    train_idx, val_idx = pblm.samples.subsplit_indices(learn_idx, n_splits=10)[0]

    # Split everything in the learning set into training / validation
    train_dataset = load_dataset(train_idx)
    vali_dataset = load_dataset(val_idx)
    test_dataset = load_dataset(test_idx)

    print('* len(train_dataset) = {}'.format(len(train_dataset)))
    print('* len(vali_dataset) = {}'.format(len(vali_dataset)))
    print('* len(test_dataset) = {}'.format(len(test_dataset)))

    from wbia.algo.verif.torch import gpu_util

    gpu_num = gpu_util.find_unused_gpu(min_memory=6000)

    use_cuda = gpu_num is not None
    data_kw = {}
    if use_cuda:
        data_kw = {'num_workers': 6, 'pin_memory': True}
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **data_kw
    )
    vali_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **data_kw
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **data_kw
    )

    from wbia.algo.verif.torch import fit_harness
    from wbia.algo.verif.torch import models
    from wbia.algo.verif.torch import netmath
    from wbia.algo.verif.torch import lr_schedule

    model = models.Siamese()

    criterion = netmath.Criterions.ContrastiveLoss(margin=1)
    lr_scheduler = lr_schedule.Exponential()
    optimizer_cls = netmath.Optimizers.Adam

    class_weights = train_dataset.class_weights()
    print('class_weights = {!r}'.format(class_weights))

    harn = fit_harness.FitHarness(
        model=model,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        vali_loader=vali_loader,
        test_loader=test_loader,
        optimizer_cls=optimizer_cls,
        class_weights=class_weights,
        gpu_num=gpu_num,
    )
    harn.run()


class LabeledPairDataset(torch.utils.data.Dataset):
    """
    transform=transforms.Compose([
                       transforms.Scale(224),
                       transforms.ToTensor(),
                       torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.225, 0.225, 0.225])
                   ]

    Ignore:
        >>> from wbia.algo.verif.torch.train_main import *
        >>> from wbia.algo.verif.vsone import *  # NOQA
        >>> pblm = OneVsOneProblem.from_empty('PZ_MTEST')
        >>> ibs = pblm.infr.ibs
        >>> pblm.load_samples()
        >>> samples = pblm.samples
        >>> samples.print_info()
        >>> xval_kw = pblm.xval_kw.asdict()
        >>> skf_list = pblm.samples.stratified_kfold_indices(**xval_kw)
        >>> train_idx, test_idx = skf_list[0]
        >>> aids1, aids2 = pblm.samples.aid_pairs[train_idx].T
        >>> labels = pblm.samples['match_state'].y_enc[train_idx]
        >>> labels = (labels == 1).astype(np.int64)
        >>> chip_config = {'resize_dim': 'wh', 'dim_size': (224, 224)}
        >>> img1_fpaths = ibs.depc_annot.get('chips', aids1, read_extern=False, colnames='img', config=chip_config)
        >>> img2_fpaths = ibs.depc_annot.get('chips', aids2, read_extern=False, colnames='img', config=chip_config)
        >>> self = LabeledPairDataset(img1_fpaths, img2_fpaths, labels)
        >>> img1, img2, label = self[0]
    """

    def __init__(self, img1_fpaths, img2_fpaths, labels, transform='default'):
        assert len(img1_fpaths) == len(img2_fpaths)
        assert len(labels) == len(img2_fpaths)
        self.img1_fpaths = img1_fpaths
        self.img2_fpaths = img2_fpaths
        self.labels = labels
        if transform == 'default':
            transform = torchvision.transforms.Compose(
                [
                    # torchvision.transforms.Scale(224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.225, 0.225, 0.225]
                    ),
                ]
            )
        self.transform = transform

    def class_weights(self):
        import pandas as pd

        label_freq = pd.value_counts(self.labels)
        class_weights = label_freq.median() / label_freq
        class_weights = class_weights.sort_index().values
        class_weights = torch.from_numpy(class_weights.astype(np.float32))
        return class_weights

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image1, image2, label)
        """
        fpath1 = self.img1_fpaths[index]
        fpath2 = self.img2_fpaths[index]
        label = self.labels[index]

        def loader(fpath):
            bgr_255 = cv2.imread(fpath)
            bgr_01 = bgr_255.astype(np.float32) / 255.0
            rgb_01 = cv2.cvtColor(bgr_01, cv2.COLOR_BGR2RGB)
            return rgb_01

        img1 = loader(fpath1)
        img2 = loader(fpath2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def __len__(self):
        return len(self.img1_fpaths)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.verif.torch.train_main
    """
    import xdoctest

    xdoctest.doctest_module(__file__)
