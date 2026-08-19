from mmengine.hooks import Hook

from mmrotate.registry import HOOKS


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


@HOOKS.register_module()
class ProgressiveEMAHook(Hook):
    """Initialize and update the experimental Point2RBox-v3 EMA teacher."""

    priority = 'LOW'

    def __init__(self, start_epoch=6, momentum=0.999):
        if start_epoch < 0:
            raise ValueError('start_epoch must be non-negative')
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum must be in [0, 1]')
        self.start_epoch = start_epoch
        self.momentum = momentum

    def before_train_epoch(self, runner):
        model = unwrap_model(runner.model)
        if runner.epoch < self.start_epoch:
            return
        if not getattr(model, 'ema_teacher_enabled', False):
            return
        if not model.teacher_initialized.item():
            model.initialize_teacher()

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch=None,
                         outputs=None):
        model = unwrap_model(runner.model)
        if runner.epoch < self.start_epoch:
            return
        if not getattr(model, 'ema_teacher_enabled', False):
            return
        if model.teacher_initialized.item():
            model.update_teacher(self.momentum)
