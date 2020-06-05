# -*- coding: utf-8 -*-
# class FitHarnessOld(object):
#     def __init__(harn, train_loader, vali_loader, test_loader):
#         harn.train_loader = train_loader
#         harn.vali_loader = vali_loader
#         harn.test_loader = test_loader
#         harn.criterion = ContrastiveLoss(margin=1.0)
#         harn.model = Siamese()
#         harn.lr_scheduler = LRSchedule.exp
#         harn.use_cuda = False
#         # harn.model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
#         harn.config = {
#             'maxIterations': 10000,
#             'displayInterval': 1,
#             'vail_displayInterval': 1,
#             'model_dir': '.',
#             'margin': 1.0,
#         }
#         harn.lr = 0.001
#         harn.epoch = 0

#     def log(harn, msg):
#         print(msg)

#     def log_value(harn, key, value, n_iter):
#         print('{}={} @ {}'.format(key, value, n_iter))

#     def load_snapshot(harn, load_path):
#         snapshot = torch.load(load_path)
#         # loadModelState(model, snapshot)
#         harn.model.load_state_dict(snapshot['state_dict'])
#         harn.epoch = snapshot['epoch'] + 1
#         harn.log('Model loaded from {}'.format(load_path))

#     def run(harn):
#         # optimizer = harn.config.optimizer(model.parameters(), lr=lr)
#         # harn.optimizer = torch.optim.SGD(harn.model.parameters(), lr=harn.lr)
#         harn.optimizer = torch.optim.Adam(harn.model.parameters(), lr=harn.lr)

#         # train loop
#         # configure("runs/afrl", flush_secs=2)

#         while True:
#             harn.train_epoch()
#             harn.validation_epoch()
#             harn.save_snapshot()

#             # check for termination
#             if harn.epoch > harn.config['maxIterations']:
#                 harn.log('Maximum harn.epoch reached, terminating ...')
#                 break
#             harn.epoch += 1

#     def train_epoch(harn):
#         ave_metrics = {
#             'loss': 0,
#             'accuracy': 0,
#             'pos_dist': 0,
#             'neg_dist': 0,
#         }

#         # change learning rate
#         harn.optimizer, harn.lr = harn.lr_scheduler(harn.optimizer, harn.epoch, harn.lr, 2)

#         # train batch
#         for batch_idx, (data0, data1, target) in enumerate(harn.train_loader):
#             target = target.type(torch.FloatTensor)
#             if harn.use_cuda:
#                 data0, data1, target = data0.cuda(), data1.cuda(), target.cuda()
#             data0, data1, target = Variable(data0), Variable(data1), Variable(target)
#             input_batch = (data0, data1, target)
#             print('Begin batch {}'.format(batch_idx))
#             t_cur_metrics = harn.train_batch(input_batch)

#             for k, v in t_cur_metrics.items():
#                 ave_metrics[k] += v

#             # display training info
#             if (batch_idx + 1) % harn.config['displayInterval'] == 0:
#                 for k in ave_metrics.keys():
#                     ave_metrics[k] /= harn.config['displayInterval']

#                 n_train = len(harn.train_loader)
#                 harn.log('Epoch {0}: {1} / {2} | lr:{3} - tloss:{4:.5f} acc:{5:.2f} | sdis:{6:.3f} ddis:{7:.3f}'.format(
#                     harn.epoch, batch_idx, n_train, harn.lr,
#                     ave_metrics['loss'], ave_metrics['accuracy'],
#                     ave_metrics['pos_dist'], ave_metrics['neg_dist']))

#                 iter_idx = harn.epoch * n_train + batch_idx
#                 for key, value in ave_metrics.items():
#                     harn.log_value('train ' + key, value, iter_idx)

#                 # diagnoseGradients(model.parameters())
#                 for k in ave_metrics.keys():
#                     ave_metrics[k] = 0

#     def validation_epoch(harn):
#         ave_metrics = {
#             'loss': 0,
#             'accuracy': 0,
#             'pos_dist': 0,
#             'neg_dist': 0,
#         }

#         final_metrics = ave_metrics.copy()

#         for vali_idx, (t_data0, t_data1, t_target) in enumerate(harn.vali_loader):
#             t_target = t_target.type(torch.FloatTensor)
#             if harn.use_cuda:
#                 t_data0, t_data1, t_target = t_data0.cuda(), t_data1.cuda(), t_target.cuda()
#             t_data0, t_data1, t_target = Variable(t_data0), Variable(t_data1), Variable(t_target)

#             input_batch = (t_data0, t_data1, t_target)
#             print('Begin batch {}'.format(vali_idx))
#             v_cur_metrics = harn.validation_batch(input_batch)

#             for k, v in v_cur_metrics.items():
#                 ave_metrics[k] += v
#                 final_metrics[k] += v

#             if (vali_idx + 1) % harn.config['vail_displayInterval'] == 0:
#                 for k in ave_metrics.keys():
#                     ave_metrics[k] /= harn.config['displayInterval']

#                 harn.log('Epoch {0}: {1} / {2} | vloss:{3:.5f} acc:{4:.2f} | sdis:{5:.3f} ddis:{6:.3f}'.format(
#                     harn.epoch, vali_idx, len(harn.vali_loader),
#                     ave_metrics['loss'], ave_metrics['accuracy'],
#                     ave_metrics['pos_dist'], ave_metrics['neg_dist']))

#                 for k in ave_metrics.keys():
#                     ave_metrics[k] = 0

#         for k in final_metrics.keys():
#             final_metrics[k] /= len(harn.vali_loader)
#         harn.log('Epoch {0}: final vloss:{1:.5f} acc:{2:.2f} | sdis:{3:.3f} ddis:{4:.3f}'.format(
#             harn.epoch, final_metrics['loss'], final_metrics['accuracy'],
#             final_metrics['pos_dist'], final_metrics['neg_dist']))

#         iter_idx = harn.epoch * len(harn.vali_loader) + vali_idx
#         for key, value in final_metrics.items():
#             harn.log_value('validation ' + key, value, iter_idx)

#     def save_snapshot(harn):
#         # save snapshot
#         save_path = join(harn.config['model_dir'], 'snapshot_epoch_{}.pt'.format(harn.epoch))
#         # torch.save(checkpoint(model, harn.epoch), save_path)
#         harn.log('Snapshot saved to {}'.format(save_path))

#     def train_batch(harn, input_batch):
#         harn.model.train(True)
#         input1, input2, label = input_batch
#         output0, output1, output = harn.model(input1, input2)
#         t_metrics = harn._measure_metrics(output0, output1, label, harn.config['margin'])

#         # loss = harn.criterion(output, label)
#         loss = harn.criterion(output0, output1, label)

#         harn.optimizer.zero_grad()
#         loss.backward()
#         harn.optimizer.step()

#         # loss = loss / input1.size()[0]

#         loss_sum = loss.data.sum()

#         inf = float("inf")
#         if loss_sum == inf or loss_sum == -inf:
#             harn.log("WARNING: received an inf loss, setting loss value to 0")
#             loss_value = 0
#         else:
#             loss_value = loss.data[0]

#         t_metrics['loss'] = loss_value
#         return t_metrics

#     def validation_batch(harn, input_batch):
#         harn.model.train(False)
#         input1, input2, label = input_batch

#         output0, output1, output = harn.model(input1, input2)
#         v_metrics = harn._measure_metrics(output0, output1, label, harn.config['margin'])

#         # loss = harn.criterion(output, label)
#         loss = harn.criterion(output0, output1, label)
#         # loss = loss / input1.size()[0]
#         loss_sum = loss.data.sum()

#         inf = float("inf")
#         if loss_sum == inf or loss_sum == -inf:
#             harn.log("WARNING: received an inf loss, setting loss value to 0")
#             loss_value = 0
#         else:
#             loss_value = loss.data[0]

#         v_metrics['loss'] = loss_value
#         return v_metrics

#     def _measure_metrics(harn, output0, output1, label, margin):
#         diff = torch.abs(output0 - output1)
#         l21 = torch.sqrt(torch.pow(diff, 2).sum(dim=1))

#         label_tensor = torch.from_numpy(label.data.cpu().numpy())
#         l21_tensor = torch.from_numpy(l21.data.cpu().numpy())

#         # Distance
#         is_pos = torch.ByteTensor()
#         POS_LABEL = 1
#         NEG_LABEL = 0
#         torch.eq(label_tensor, POS_LABEL, out=is_pos)  # y==1
#         pos_dist = 0 if len(l21_tensor[is_pos]) == 0 else l21_tensor[is_pos].mean()
#         neg_dist = 0 if len(l21_tensor[~is_pos]) == 0 else l21_tensor[~is_pos].mean()
#         # print('same dis : diff dis  {} : {}'.format(l21_tensor[is_pos == 0].mean(), l21_tensor[is_pos].mean()))

#         # accuracy
#         pred_pos_flags = torch.ByteTensor()
#         torch.le(l21_tensor, margin, out=pred_pos_flags)  # y==1's idx

#         cur_score = torch.FloatTensor(label.size(0))
#         cur_score.fill_(NEG_LABEL)
#         cur_score[pred_pos_flags] = POS_LABEL

#         accuracy = torch.eq(cur_score, label_tensor).sum() / label_tensor.size(0)

#         metrics = {
#             'accuracy': accuracy,
#             'pos_dist': pos_dist,
#             'neg_dist': neg_dist,
#         }

#         return metrics
