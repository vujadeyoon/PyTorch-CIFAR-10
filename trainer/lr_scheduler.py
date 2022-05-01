import torch


class LRScheduler(object):
    def __init__(self, _config, _optimizer, _warmup_alogrithm: str, _warmup_epoch: int, _warmup_lr: float) -> None:
        super(LRScheduler, self).__init__()
        self.config = _config
        self.optimizer = _optimizer
        self.warmup_epoch = _warmup_epoch
        self.warmup_lr = _warmup_lr
        self.warmup_alogrithm = _warmup_alogrithm
        self.lr = self.get_lr()
        self.lr_scheduler = self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
        self.idx_epoch = 1 # it is assumed that the number of _epoch should be start wtih 1, not 0.

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def step(self) -> None:
        if self.idx_epoch <= self.warmup_epoch:
            self._adjust_lr(_lr=self._get_lr_warmup())
        else:
            if 1 < self.idx_epoch:
                self.lr_scheduler.step()

        self.idx_epoch += 1

    def _get_lr_warmup(self) -> float:
        if self.warmup_alogrithm == 'Multiplicative':
            warmup_factor = (self.lr / self.warmup_lr) ** (1.0 / (self.warmup_epoch - 1))
            res = self.warmup_lr * (warmup_factor ** (self.idx_epoch - 1))
        else:
            raise NotImplementedError

        return res

    def _adjust_lr(self, _lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = _lr
