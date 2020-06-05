# -*- coding: utf-8 -*-
class Exponential(object):
    """
    Decay learning rate by a factor of `decay_rate` every `lr_decay_epoch`
    epochs.

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.verif.torch.lr_schedule import *
        >>> lr_scheduler = Exponential()
        >>> rates = np.array([lr_scheduler(i) for i in range(6)])
        >>> target = np.array([1E-3, 1E-3, 1E-5, 1E-5, 1E-7, 1E-7])
        >>> assert all(list(np.isclose(target, rates)))
    """

    def __init__(self, init_lr=0.001, decay_rate=0.01, lr_decay_epoch=100):
        self.init_lr = init_lr
        self.decay_rate = 0.01
        self.lr_decay_epoch = lr_decay_epoch

    def __call__(self, epoch, optimizer=None):
        """
        If optimizer is specified, its learning rate is modified inplace.
        """
        n_decays = epoch // self.lr_decay_epoch
        lr = self.init_lr * (self.decay_rate ** n_decays)

        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        return lr
