# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
import cv2
import numpy as np
import vtool as vt
import torch
import torch.nn
import utool as ut
import torchvision
from torch.autograd import Variable
print, rrr, profile = ut.inject2(__name__)


def testdata_siam_desc(num_data=128, desc_dim=8):
    rng = np.random.RandomState(0)
    network_output = vt.normalize_rows(rng.rand(num_data, desc_dim))
    vecs1 = network_output[0::2]
    vecs2 = network_output[1::2]
    # roll vecs2 so it is essentially translated
    vecs2 = np.roll(vecs1, 1, axis=1)
    network_output[1::2] = vecs2
    # Every other pair is an imposter match
    network_output[::4, :] = vt.normalize_rows(rng.rand(32, desc_dim))
    #data_per_label = 2

    vecs1 = network_output[0::2].astype(np.float32)
    vecs2 = network_output[1::2].astype(np.float32)

    def true_dist_metric(vecs1, vecs2):
        g1_ = np.roll(vecs1, 1, axis=1)
        dist = vt.L2(g1_, vecs2)
        return dist
    #l2dist = vt.L2(vecs1, vecs2)
    true_dist = true_dist_metric(vecs1, vecs2)
    labels = (true_dist > 0).astype(np.float32)
    vecs1 = torch.from_numpy(vecs1)
    vecs2 = torch.from_numpy(vecs2)
    labels = torch.from_numpy(labels)
    return vecs1, vecs2, labels


class LRSchedule(object):
    @staticmethod
    def exp(optimizer, epoch, init_lr=0.001, lr_decay_epoch=2):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr
        # epoch += 1
        if epoch % lr_decay_epoch == 0 and epoch is not 0:
            lr *= 0.1

        if epoch % lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer, lr


def siam_vsone_problem():
    # wrapper around the RF vsone problem
    from ibeis.algo.verif import vsone
    pblm = vsone.OneVsOneProblem.from_empty('PZ_MTEST')
    ibs = pblm.infr.ibs
    pblm.load_samples()
    samples = pblm.samples
    samples.print_info()
    xval_kw = pblm.xval_kw.asdict()
    skf_list = pblm.samples.stratified_kfold_indices(**xval_kw)

    def load_dataset(subset_idx):
        aids1, aids2 = pblm.samples.aid_pairs[subset_idx].T
        labels = pblm.samples['match_state'].y_enc[subset_idx]
        chip_config = {'resize_dim': 'wh', 'dim_size': (224, 224)}
        img1_fpaths = ibs.depc_annot.get('chips', aids1, read_extern=False, colnames='img', config=chip_config)
        img2_fpaths = ibs.depc_annot.get('chips', aids2, read_extern=False, colnames='img', config=chip_config)
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

    use_cuda = False
    data_kw = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                               shuffle=True, **data_kw)
    vali_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                              shuffle=True, **data_kw)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                              shuffle=False, **data_kw)

    harn = FitHarness(train_loader, vali_loader, test_loader)
    harn.run()


class FitHarness(object):
    def __init__(harn, train_loader, vali_loader, test_loader):
        harn.train_loader = train_loader
        harn.vali_loader = vali_loader
        harn.test_loader = test_loader
        harn.criterion = ContrastiveLoss(margin=1.0)
        harn.model = Siamese()
        harn.lr_scheduler = LRSchedule.exp
        harn.use_cuda = False
        # harn.model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
        harn.config = {
            'maxIterations': 10000,
            'displayInterval': 1,
            'vail_displayInterval': 1,
            'model_dir': '.',
            'margin': 1.0,
        }
        harn.lr = 0.001
        harn.epoch = 0

    def log(harn, msg):
        print(msg)

    def log_value(harn, key, value, n_iter):
        print('{}={} @ {}'.format(key, value, n_iter))

    def load_snapshot(harn, load_path):
        snapshot = torch.load(load_path)
        # loadModelState(model, snapshot)
        harn.model.load_state_dict(snapshot['state_dict'])
        harn.epoch = snapshot['epoch'] + 1
        harn.log('Model loaded from {}'.format(load_path))

    def run(harn):
        # optimizer = harn.config.optimizer(model.parameters(), lr=lr)
        # harn.optimizer = torch.optim.SGD(harn.model.parameters(), lr=harn.lr)
        harn.optimizer = torch.optim.Adam(harn.model.parameters(), lr=harn.lr)

        # train loop
        # configure("runs/afrl", flush_secs=2)

        while True:
            harn.train_epoch()
            harn.validation_epoch()
            harn.save_snapshot()

            # check for termination
            if harn.epoch > harn.config['maxIterations']:
                harn.log('Maximum harn.epoch reached, terminating ...')
                break
            harn.epoch += 1

    def train_epoch(harn):
        ave_metrics = {
            'loss': 0,
            'accuracy': 0,
            'pos_dist': 0,
            'neg_dist': 0,
        }

        # change learning rate
        harn.optimizer, harn.lr = harn.lr_scheduler(harn.optimizer, harn.epoch, harn.lr, 2)

        # train batch
        for batch_idx, (data0, data1, target) in enumerate(harn.train_loader):
            target = target.type(torch.FloatTensor)
            if harn.use_cuda:
                data0, data1, target = data0.cuda(), data1.cuda(), target.cuda()
            data0, data1, target = Variable(data0), Variable(data1), Variable(target)
            input_batch = (data0, data1, target)
            print('Begin batch {}'.format(batch_idx))
            t_cur_metrics = harn.train_batch(input_batch)

            for k, v in t_cur_metrics.items():
                ave_metrics[k] += v

            # display training info
            if (batch_idx + 1) % harn.config['displayInterval'] == 0:
                for k in ave_metrics.keys():
                    ave_metrics[k] /= harn.config['displayInterval']

                n_train = len(harn.train_loader)
                harn.log('Epoch {0}: {1} / {2} | lr:{3} - tloss:{4:.5f} acc:{5:.2f} | sdis:{6:.3f} ddis:{7:.3f}'.format(
                    harn.epoch, batch_idx, n_train, harn.lr,
                    ave_metrics['loss'], ave_metrics['accuracy'],
                    ave_metrics['pos_dist'], ave_metrics['neg_dist']))

                iter_idx = harn.epoch * n_train + batch_idx
                for key, value in ave_metrics.items():
                    harn.log_value('train ' + key, value, iter_idx)

                # diagnoseGradients(model.parameters())
                for k in ave_metrics.keys():
                    ave_metrics[k] = 0

    def validation_epoch(harn):
        ave_metrics = {
            'loss': 0,
            'accuracy': 0,
            'pos_dist': 0,
            'neg_dist': 0,
        }

        final_metrics = ave_metrics.copy()

        for vali_idx, (t_data0, t_data1, t_target) in enumerate(harn.vali_loader):
            t_target = t_target.type(torch.FloatTensor)
            if harn.use_cuda:
                t_data0, t_data1, t_target = t_data0.cuda(), t_data1.cuda(), t_target.cuda()
            t_data0, t_data1, t_target = Variable(t_data0), Variable(t_data1), Variable(t_target)

            input_batch = (t_data0, t_data1, t_target)
            print('Begin batch {}'.format(vali_idx))
            v_cur_metrics = harn.validation_batch(input_batch)

            for k, v in v_cur_metrics.items():
                ave_metrics[k] += v
                final_metrics[k] += v

            if (vali_idx + 1) % harn.config['vail_displayInterval'] == 0:
                for k in ave_metrics.keys():
                    ave_metrics[k] /= harn.config['displayInterval']

                harn.log('Epoch {0}: {1} / {2} | vloss:{3:.5f} acc:{4:.2f} | sdis:{5:.3f} ddis:{6:.3f}'.format(
                    harn.epoch, vali_idx, len(harn.vali_loader),
                    ave_metrics['loss'], ave_metrics['accuracy'],
                    ave_metrics['pos_dist'], ave_metrics['neg_dist']))

                for k in ave_metrics.keys():
                    ave_metrics[k] = 0

        for k in final_metrics.keys():
            final_metrics[k] /= len(harn.vali_loader)
        harn.log('Epoch {0}: final vloss:{1:.5f} acc:{2:.2f} | sdis:{3:.3f} ddis:{4:.3f}'.format(
            harn.epoch, final_metrics['loss'], final_metrics['accuracy'],
            final_metrics['pos_dist'], final_metrics['neg_dist']))

        iter_idx = harn.epoch * len(harn.vali_loader) + vali_idx
        for key, value in final_metrics.items():
            harn.log_value('validation ' + key, value, iter_idx)

    def save_snapshot(harn):
        # save snapshot
        save_path = join(harn.config['model_dir'], 'snapshot_epoch_{}.pt'.format(harn.epoch))
        # torch.save(checkpoint(model, harn.epoch), save_path)
        harn.log('Snapshot saved to {}'.format(save_path))

    def train_batch(harn, input_batch):
        harn.model.train(True)
        input1, input2, label = input_batch
        output0, output1, output = harn.model(input1, input2)
        t_metrics = harn._measure_metrics(output0, output1, label, harn.config['margin'])

        # loss = harn.criterion(output, label)
        loss = harn.criterion(output0, output1, label)

        harn.optimizer.zero_grad()
        loss.backward()
        harn.optimizer.step()

        # loss = loss / input1.size()[0]

        loss_sum = loss.data.sum()

        inf = float("inf")
        if loss_sum == inf or loss_sum == -inf:
            harn.log("WARNING: received an inf loss, setting loss value to 0")
            loss_value = 0
        else:
            loss_value = loss.data[0]

        t_metrics['loss'] = loss_value
        return t_metrics

    def validation_batch(harn, input_batch):
        harn.model.train(False)
        input1, input2, label = input_batch

        output0, output1, output = harn.model(input1, input2)
        v_metrics = harn._measure_metrics(output0, output1, label, harn.config['margin'])

        # loss = harn.criterion(output, label)
        loss = harn.criterion(output0, output1, label)
        # loss = loss / input1.size()[0]
        loss_sum = loss.data.sum()

        inf = float("inf")
        if loss_sum == inf or loss_sum == -inf:
            harn.log("WARNING: received an inf loss, setting loss value to 0")
            loss_value = 0
        else:
            loss_value = loss.data[0]

        v_metrics['loss'] = loss_value
        return v_metrics

    def _measure_metrics(harn, output0, output1, label, margin):
        diff = torch.abs(output0 - output1)
        l21 = torch.sqrt(torch.pow(diff, 2).sum(dim=1))

        label_tensor = torch.from_numpy(label.data.cpu().numpy())
        l21_tensor = torch.from_numpy(l21.data.cpu().numpy())

        # Distance
        is_pos = torch.ByteTensor()
        POS_LABEL = 1
        NEG_LABEL = 0
        torch.eq(label_tensor, POS_LABEL, out=is_pos)  # y==1
        pos_dist = 0 if len(l21_tensor[is_pos]) == 0 else l21_tensor[is_pos].mean()
        neg_dist = 0 if len(l21_tensor[~is_pos]) == 0 else l21_tensor[~is_pos].mean()
        # print('same dis : diff dis  {} : {}'.format(l21_tensor[is_pos == 0].mean(), l21_tensor[is_pos].mean()))

        # accuracy
        pred_pos_flags = torch.ByteTensor()
        torch.le(l21_tensor, margin, out=pred_pos_flags)  # y==1's idx

        cur_score = torch.FloatTensor(label.size(0))
        cur_score.fill_(NEG_LABEL)
        cur_score[pred_pos_flags] = POS_LABEL

        accuracy = torch.eq(cur_score, label_tensor).sum() / label_tensor.size(0)

        metrics = {
            'accuracy': accuracy,
            'pos_dist': pos_dist,
            'neg_dist': neg_dist,
        }

        return metrics


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.

    References:
        https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py

    LaTeX:
        $(y E)^2 + ((1 - y) max(m - E, 0)^2)$

    Example:
        >>> from ibeis.algo.verif.siamese import *
        >>> vecs1, vecs2, labels = testdata_siam_desc()
        >>> self = ContrastiveLoss()
        >>> ut.exec_func_src(self.forward, globals())
        >>> func = self.forward
        >>> loss2x, dist_l2 = ut.exec_func_src(self.forward, globals(), globals(), keys=['loss2x', 'dist_l2'])
        >>> ut.quit_if_noshow()
        >>> loss2x, dist_l2, labels = map(np.array, [loss, dist_l2, labels])
        >>> labels = labels.astype(np.bool)
        >>> dist0_l2 = dist_l2[labels]
        >>> dist1_l2 = dist_l2[~labels]
        >>> loss0 = loss2x[labels] / 2
        >>> loss1 = loss2x[~labels] / 2
        >>> import plottool as pt
        >>> pt.plot2(dist0_l2, loss0, 'x', color=pt.TRUE_BLUE, label='imposter_loss', y_label='loss')
        >>> pt.plot2(dist1_l2, loss1, 'x', color=pt.FALSE_RED, label='genuine_loss', y_label='loss')
        >>> pt.gca().set_xlabel('l2-dist')
        >>> pt.legend()
        >>> ut.show_if_requested()
    """

    def __init__(self, margin=1.0):
        ut.super2(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, vecs1, vecs2, labels):
        # euclidian distance
        diff = vecs1 - vecs2
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist_l2 = torch.sqrt(dist_sq)

        loss2x_genuine  = (1 - labels) * torch.pow(torch.clamp(self.margin - dist_l2, min=0.0), 2)
        loss2x_imposter = labels * dist_sq
        loss2x = loss2x_genuine + loss2x_imposter
        ave_loss = torch.sum(loss2x) / 2.0 / vecs1.size()[0]
        return ave_loss


class LabeledPairDataset(torch.utils.data.Dataset):
    """
    transform=transforms.Compose([
                       transforms.Scale(224),
                       transforms.ToTensor(),
                       torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.225, 0.225, 0.225])
                   ]

    Ignore:
        >>> from ibeis.algo.verif.siamese import *
        >>> from ibeis.algo.verif.vsone import *  # NOQA
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
        if transform  == 'default':
            transform = torchvision.transforms.Compose([
                # torchvision.transforms.Scale(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5],
                                                 [0.225, 0.225, 0.225]),
            ])
        self.transform = transform

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


class Siamese(torch.nn.Module):
    """
    Example:
        >>> from ibeis.algo.verif.siamese import *
        >>> self = Siamese()
    """

    def __init__(self):
        ut.super2(Siamese, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.num_fcin = self.resnet.fc.in_features
        # replace the last layer of resnet
        self.resnet.fc = torch.nn.Linear(self.num_fcin, 500)
        self.pdist = torch.nn.PairwiseDistance(1)

    def forward(self, input1, input2):
        output1 = self.resnet(input1)
        output2 = self.resnet(input2)
        output = self.pdist(output1, output2)

        return output1, output2, output


def visualize():
    import networkx as nx
    import torch
    from torch.autograd import Variable

    def make_nx(var, params):
        param_map = {id(v): k for k, v in params.items()}
        print(param_map)
        node_attr = dict(style='filled', shape='box', align='left',
                         fontsize='12', ranksep='0.1', height='0.2')
        seen = set()
        G = nx.DiGraph()

        def size_to_str(size):
            return '(' + (', ').join(['%d' % v for v in size]) + ')'

        def build_graph(var):
            if var not in seen:
                if torch.is_tensor(var):
                    G.add_node(id(var), label=size_to_str(var.size()),
                               fillcolor='orange', **node_attr)
                elif hasattr(var, 'variable'):
                    u = var.variable
                    node_name = '%s\n %s' % (param_map.get(id(u)),
                                             size_to_str(u.size()))
                    G.add_node(id(var), label=node_name,
                               fillcolor='lightblue', **node_attr)
                else:
                    G.add_node(id(var), label=str(type(var).__name__),
                               **node_attr)
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            G.add_edge(id(u[0]), id(var))
                            build_graph(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        G.add_edge(id(t), id(var))
                        build_graph(t)
        build_graph(var.grad_fn)
        return G

    # inputs = torch.randn(1, 3, 224, 224)
    # resnet18 = models.resnet18()
    # y = resnet18(Variable(inputs))

    inputs = torch.randn(1, 3, 224, 224)
    # model = torchvision.models.resnet18()
    model = torchvision.models.resnet50()

    model = Siamese()

    # y = model(Variable(inputs))
    y = model(Variable(inputs), Variable(inputs))

    params = model.state_dict()
    G = make_nx(y, params)

    import plottool as pt
    pt.dump_nx_ondisk(G, './pytorch_network.png')
    ut.startfile( './pytorch_network.png')
    # pt.show_nx(G, arrow_width=1)
    # pt.zoom_factory()
    # pt.pan_factory()
