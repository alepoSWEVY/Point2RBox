from types import SimpleNamespace

import torch
from torch import nn

from projects.point2rbox_v3_soft_pgdm.ema_detector import EMATeacherMixin
from projects.point2rbox_v3_soft_pgdm.progressive_ema_hook import (
    ProgressiveEMAHook)


class ToyModule(nn.Module):

    def __init__(self, value):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value], dtype=torch.float32))
        self.register_buffer('running', torch.tensor([value]))
        self.register_buffer('counter', torch.tensor(0, dtype=torch.long))


class ToyDetector(EMATeacherMixin, nn.Module):

    def __init__(self, enabled=True):
        super().__init__()
        self.backbone = ToyModule(1.0)
        self.neck = ToyModule(1.0)
        self.bbox_head = ToyModule(1.0)
        self._init_ema_teacher(dict(enabled=enabled))

    @property
    def with_neck(self):
        return self.neck is not None


def test_ema_numerics_and_integer_buffers():
    model = ToyDetector()
    model.initialize_teacher()
    for _, student in model._teacher_student_pairs():
        student.weight.data.fill_(3.0)
        student.running.fill_(3.0)
        student.counter.fill_(7)

    model.update_teacher(0.9)

    for teacher, _ in model._teacher_student_pairs():
        assert torch.allclose(teacher.weight, torch.tensor([1.2]))
        assert torch.allclose(teacher.running, torch.tensor([1.2]))
        assert teacher.counter.item() == 7
    assert model.teacher_update_steps.item() == 1


def test_teacher_is_frozen_and_stays_in_eval_mode():
    model = ToyDetector()
    model.train()
    for teacher, _ in model._teacher_student_pairs():
        assert not teacher.training
        assert all(not parameter.requires_grad
                   for parameter in teacher.parameters())
        assert all(parameter.grad is None for parameter in teacher.parameters())


def test_progressive_hook_start_epoch_and_wrapped_model():
    model = ToyDetector()
    wrapped = SimpleNamespace(module=model)
    runner = SimpleNamespace(model=wrapped, epoch=5)
    hook = ProgressiveEMAHook(start_epoch=6, momentum=0.9)

    hook.before_train_epoch(runner)
    hook.after_train_iter(runner, 0)
    assert not model.teacher_initialized.item()

    runner.epoch = 6
    hook.before_train_epoch(runner)
    assert model.teacher_initialized.item()
    hook.after_train_iter(runner, 0)
    assert model.teacher_update_steps.item() == 1


def test_teacher_checkpoint_state_is_not_reinitialized():
    model = ToyDetector()
    model.initialize_teacher()
    model.teacher_backbone.weight.data.fill_(5.0)
    model.teacher_update_steps.fill_(9)
    state_dict = model.state_dict()

    resumed = ToyDetector()
    resumed.load_state_dict(state_dict, strict=True)
    runner = SimpleNamespace(model=resumed, epoch=8)
    ProgressiveEMAHook(start_epoch=6).before_train_epoch(runner)

    assert resumed.teacher_initialized.item()
    assert resumed.teacher_update_steps.item() == 9
    assert torch.allclose(resumed.teacher_backbone.weight,
                          torch.tensor([5.0]))
