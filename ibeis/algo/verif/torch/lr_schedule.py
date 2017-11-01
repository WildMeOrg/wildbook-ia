
class Exponential(object):
    def __init__(self, init_lr=0.001, lr_decay_epoch=2):
        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch

    def __call__(self, epoch, optimizer):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = self.init_lr
        if epoch % self.lr_decay_epoch == 0 and epoch is not 0:
            lr *= 0.1
        if epoch % self.lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
