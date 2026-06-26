'''
This file contains a class which wraps the scheduler to create a unifrom interace for interacting 
with the scheduler. This may be a little overengineered.
'''
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SchedulerWrapper:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._is_plateau = isinstance(scheduler, ReduceLROnPlateau)

    def step(self, val_loss=None):
        if self._is_plateau:
            if val_loss is None:
                raise ValueError("ReduceLROnPlateau requires val_loss")
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def get_last_lr(self):
        # plateau lacks this on older torch; fall back to the optimizer
        if hasattr(self.scheduler, "get_last_lr"):
            return self.scheduler.get_last_lr()
        return [g["lr"] for g in self.scheduler.optimizer.param_groups]

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, sd):
        self.scheduler.load_state_dict(sd)