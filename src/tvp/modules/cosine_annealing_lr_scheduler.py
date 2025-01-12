import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingLRScheduler(_LRScheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        base_lr, 
        warmup_length, 
        total_steps, 
        last_epoch=-1
    ):
        self.base_lrs = (
            [base_lr] if not isinstance(base_lr, list) else base_lr
        )
        self.warmup_length = warmup_length
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # "last_epoch" is how PyTorch LRScheduler tracks the step count
        step = self.last_epoch
        # For each param_group and base_lr, compute the new LR
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_length:
                # warmup
                lr = base_lr * (step + 1) / self.warmup_length
            else:
                e = step - self.warmup_length
                es = self.total_steps - self.warmup_length
                lr = 0.5 * (1 + math.cos(math.pi * e / es)) * base_lr
            lrs.append(lr)
        return lrs
