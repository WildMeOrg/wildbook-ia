# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import torch
from collections import defaultdict
from os.path import join
from torch.autograd import Variable
from wbia.algo.verif.torch import netmath
import tensorboard_logger
import utool as ut

# from wbia.algo.verif.torch import gpu_util

(print, rrr, profile) = ut.inject2(__name__)


class FitHarness(object):
    def __init__(
        harn,
        model,
        train_loader,
        vali_loader=None,
        test_loader=None,
        criterion='cross_entropy',
        lr_scheduler='exp',
        optimizer_cls='Adam',
        class_weights=None,
        gpu_num=None,
        workdir=None,
    ):

        harn.workdir = workdir

        harn.train_loader = train_loader
        harn.vali_loader = vali_loader
        harn.test_loader = test_loader

        harn.model = model

        harn.optimizer_cls = optimizer_cls
        harn.criterion = criterion
        harn.lr_scheduler = lr_scheduler
        # netmath.Optimizers.lookup(optimizer_cls)
        # netmath.Criterions.lookup(criterion)
        # netmath.LRSchedules.lookup(lr_scheduler)

        harn.class_weights = class_weights

        harn.gpu_num = gpu_num
        harn.use_cuda = harn.gpu_num is not None

        # harn.model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
        harn.config = {
            'maxIterations': 10000,
            'displayInterval': 1,
            'vail_displayInterval': 1,
            'model_dir': '.',
            'margin': 1.0,
        }
        harn.lr = harn.lr_scheduler.init_lr
        harn.epoch = 0

    def log(harn, msg):
        print(msg)

    def log_value(harn, key, value, n_iter):
        if False:
            print('{}={} @ {}'.format(key, value, n_iter))
        if tensorboard_logger:
            tensorboard_logger.log_value(key, value, n_iter)

    def _to_xpu(harn, *args):
        """ Puts data on the GPU if available """
        if harn.use_cuda:
            args = [Variable(item.cuda(harn.gpu_num)) for item in args]
            # input_batch = [Variable(item.cuda()) for item in input_batch]
        else:
            args = [Variable(item) for item in args]
        return args

    def run(harn):
        harn.log('Begin training')

        if False:
            # TODO: can we run this as a subprocess that dies when we die?
            # or do we need to run externally?
            # tensorboard --logdir runs
            # http://aretha:6006
            pass

        if tensorboard_logger:
            harn.log('Initializing tensorboard')
            tensorboard_logger.configure('runs/wbia', flush_secs=2)

        if harn.use_cuda:
            harn.log('Fitting model on GPU({})'.format(harn.gpu_num))
            harn.model.cuda(harn.gpu_num)
        else:
            harn.log('Fitting model on the CPU')

        if harn.class_weights is not None:
            (harn.class_weights,) = harn._to_xpu(harn.class_weights)

        lr = harn.lr_scheduler(harn.epoch)
        harn.optimizer = harn.optimizer_cls(harn.model.parameters(), lr=lr)

        # train loop

        while not harn.check_termination():
            harn.train_epoch()

            if harn.vali_loader:
                harn.validation_epoch()

            harn.save_snapshot()

            harn.epoch += 1

    def check_termination(harn):
        # check for termination
        if harn.epoch > harn.config['maxIterations']:
            harn.log('Maximum harn.epoch reached, terminating ...')
            return True
        return False

    def train_epoch(harn):
        ave_metrics = defaultdict(lambda: 0)

        # change learning rate (modified optimizer inplace)
        lr = harn.lr_scheduler(harn.epoch, harn.optimizer)

        # train batch
        for batch_idx, input_batch in enumerate(harn.train_loader):
            input_batch = harn._to_xpu(*input_batch)

            # print('Begin batch {}'.format(batch_idx))
            t_cur_metrics = harn.train_batch(input_batch)

            for k, v in t_cur_metrics.items():
                ave_metrics[k] += v

            # display training info
            if (batch_idx + 1) % harn.config['displayInterval'] == 0:
                for k in ave_metrics.keys():
                    ave_metrics[k] /= harn.config['displayInterval']

                n_train = len(harn.train_loader)
                harn.log(
                    'Epoch {0}: {1} / {2} | lr:{3} - tloss:{4:.5f} acc:{5:.2f} | sdis:{6:.3f} ddis:{7:.3f}'.format(
                        harn.epoch,
                        batch_idx,
                        n_train,
                        lr,
                        ave_metrics['loss'],
                        ave_metrics['accuracy'],
                        ave_metrics['pos_dist'],
                        ave_metrics['neg_dist'],
                    )
                )

                iter_idx = harn.epoch * n_train + batch_idx
                for key, value in ave_metrics.items():
                    harn.log_value('train ' + key, value, iter_idx)

                # diagnoseGradients(model.parameters())
                for k in ave_metrics.keys():
                    ave_metrics[k] = 0

    def validation_epoch(harn):
        ave_metrics = defaultdict(lambda: 0)

        final_metrics = ave_metrics.copy()

        for vali_idx, input_batch in enumerate(harn.vali_loader):
            input_batch = harn._to_xpu(*input_batch)

            # print('Begin batch {}'.format(vali_idx))
            v_cur_metrics = harn.validation_batch(input_batch)

            for k, v in v_cur_metrics.items():
                ave_metrics[k] += v
                final_metrics[k] += v

            if (vali_idx + 1) % harn.config['vail_displayInterval'] == 0:
                for k in ave_metrics.keys():
                    ave_metrics[k] /= harn.config['displayInterval']

                harn.log(
                    'Epoch {0}: {1} / {2} | vloss:{3:.5f} acc:{4:.2f} | sdis:{5:.3f} ddis:{6:.3f}'.format(
                        harn.epoch,
                        vali_idx,
                        len(harn.vali_loader),
                        ave_metrics['loss'],
                        ave_metrics['accuracy'],
                        ave_metrics['pos_dist'],
                        ave_metrics['neg_dist'],
                    )
                )

                for k in ave_metrics.keys():
                    ave_metrics[k] = 0

        for k in final_metrics.keys():
            final_metrics[k] /= len(harn.vali_loader)
        harn.log(
            'Epoch {0}: final vloss:{1:.5f} acc:{2:.2f} | sdis:{3:.3f} ddis:{4:.3f}'.format(
                harn.epoch,
                final_metrics['loss'],
                final_metrics['accuracy'],
                final_metrics['pos_dist'],
                final_metrics['neg_dist'],
            )
        )

        iter_idx = harn.epoch * len(harn.vali_loader) + vali_idx
        for key, value in final_metrics.items():
            harn.log_value('validation ' + key, value, iter_idx)

    # def display_metrics():
    #     pass

    def load_snapshot(harn, load_path):
        snapshot = torch.load(load_path)
        harn.model.load_state_dict(snapshot['model_state_dict'])
        harn.epoch = snapshot['epoch']
        harn.log('Model loaded from {}'.format(load_path))

    def save_snapshot(harn):
        # save snapshot
        save_path = join(
            harn.config['model_dir'], 'snapshot_epoch_{}.pt'.format(harn.epoch)
        )
        snapshot = {
            'epoch': harn.epoch,
            'model_state_dict': harn.model.state_dict(),
        }
        torch.save(snapshot, save_path)
        harn.log('Snapshot saved to {}'.format(save_path))

    def train_batch(harn, input_batch):
        """
        https://github.com/meetshah1995/pytorch-semseg/blob/master/train.py
        """
        harn.model.train(True)
        *inputs, label = input_batch

        # Forward prop through the model
        output = harn.model(*inputs)

        # Compute the loss
        loss = harn.criterion(output, label, weight=harn.class_weights)

        # Measure train accuracy and other informative metrics
        t_metrics = harn._measure_metrics(output, label, loss)

        # Backprop and learn
        harn.optimizer.zero_grad()
        loss.backward()
        harn.optimizer.step()

        return t_metrics

    def validation_batch(harn, input_batch):
        harn.model.train(False)
        *inputs, label = input_batch

        output = harn.model(*inputs)

        loss = harn.criterion(output, label, weight=harn.class_weights)

        # Measure validation accuracy and other informative metrics
        v_metrics = harn._measure_metrics(output, label, loss)

        return v_metrics

    def _measure_metrics(harn, output, label, loss):
        metrics = netmath.Metrics._siamese_metrics(
            output, label, margin=harn.criterion.margin
        )

        assert 'loss' not in metrics, 'cannot compute loss as an extra metric'

        loss_sum = loss.data.sum()
        inf = float('inf')
        if loss_sum == inf or loss_sum == -inf:
            harn.log('WARNING: received an inf loss, setting loss value to 0')
            loss_value = 0
        else:
            loss_value = loss.data[0]

        metrics['loss'] = loss_value
        # metrics = {
        #     'tpr': netmath.Metrics.tpr(output, label)
        # }
        return metrics
