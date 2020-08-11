import math
import torch
from torch.optim.optimizer import Optimizer
from apex.fp16_utils import FP16_Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        # Check if using mixed precision training
        self.mixed_training = False
        base_optimizer = optimizer
        if isinstance(optimizer, FP16_Optimizer):
            self.mixed_training = True
            self.fp16_optimizer = optimizer
            base_optimizer = optimizer.optimizer
        # Check that optimizer param is valid
        elif not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        super(LRScheduler, self).__init__(base_optimizer, last_epoch)

    def step(self, epoch=None):
        # Set the current training step
        # ('epoch' is used to be consistent with _LRScheduler)
        if self.mixed_training:
            # The assumption is that the step will be constant
            state_dict = self.optimizer.state[self.optimizer.param_groups[0]['params'][0]]
            if 'step' in state_dict:
                self.last_epoch = state_dict['step'] + 1
            else:
                self.last_epoch = 1
        else:
            self.last_epoch = epoch if epoch is not None else self.last_epoch + 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineWarmupScheduler(LRScheduler):
    """
    Applies a warm up period to the learning rate.
    """

    def __init__(self, optimizer, warmup, total_steps, last_epoch=-1):
        self.warmup = warmup
        self.total_steps = total_steps
        super(CosineWarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        if progress < self.warmup:
            return [base_lr * progress / self.warmup for base_lr in self.base_lrs]
        else:
            return [base_lr * (0.5 * (1.0 + torch.cos(math.pi + progress))) for base_lr in self.base_lrs]


class ConstantWarmupScheduler(LRScheduler):
    """
    Applies a warm up period to the learning rate.
    """

    def __init__(self, optimizer, warmup, total_steps, last_epoch=-1):
        self.warmup = warmup
        self.total_steps = total_steps
        super(CosineWarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        if progress < self.warmup:
            return [base_lr * progress / self.warmup for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class LinearWarmUpScheduler(LRScheduler):
    """
    Applies a warm up period to the learning rate.
    """

    def __init__(self, optimizer, warmup, total_steps, last_epoch=-1):
        self.warmup = warmup
        self.total_steps = total_steps
        super(LinearWarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        if progress < self.warmup:
            return [base_lr * progress / self.warmup for base_lr in self.base_lrs]
        else:
            return [base_lr * max(( progress - 1.0)/(self.warmup - 1.0), 0.) for base_lr in self.base_lrs]


class PolyWarmUpConstScheduler(LRScheduler):
    """
    Applies a warm up period to the learning rate.
    """

    def __init__(self, optimizer, warmup, const, total_steps, degree=0.5, last_epoch=-1):
        self.warmup = warmup
        self.const = const
        self.warmup_const = warmup + const
        self.total_steps = total_steps
        self.degree = degree
        super(PolyWarmUpConstScheduler, self).__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        param_group = self.optimizer.param_groups[0]
        if 'step' in param_group:
            self.last_epoch = param_group['step'] + 1
        else:
            self.last_epoch = 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        if progress < self.warmup:
            return [base_lr * progress / self.warmup for base_lr in self.base_lrs]
        elif progress < self.warmup_const:
            return [base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr * max((1.0 - progress) / (1.0 - self.warmup_const), 0.) for base_lr in self.base_lrs]
            #return [base_lr * ((1.0 - progress) ** self.degree) for base_lr in self.base_lrs]
