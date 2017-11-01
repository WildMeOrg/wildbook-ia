
class Exponential(object):
    """
    Example:
        >>> from ibeis.algo.verif.torch.lr_schedule import *
        >>> lr_scheduler = Exponential()
        >>> lr_scheduler(0)
    """
    def __init__(self, init_lr=0.001, lr_decay_epoch=2):
        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch

    def __call__(self, epoch, optimizer=None):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = self.init_lr

        if epoch > 0 and epoch % self.lr_decay_epoch == 0:
            lr *= 0.1

        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        return lr
