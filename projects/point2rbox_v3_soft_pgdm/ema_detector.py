import copy

import torch

from mmrotate.models.detectors.point2rbox_v3 import Point2RBoxV3
from mmrotate.registry import MODELS


class EMATeacherMixin:
    """EMA mechanics shared by the experimental detector and unit tests."""

    def _init_ema_teacher(self, ema_teacher=None):
        self.ema_teacher_cfg = copy.deepcopy(ema_teacher or {})
        self.ema_teacher_enabled = self.ema_teacher_cfg.get('enabled', False)
        if not self.ema_teacher_enabled:
            return

        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_neck = copy.deepcopy(self.neck) \
            if self.with_neck else None
        self.teacher_bbox_head = copy.deepcopy(self.bbox_head)
        self.register_buffer(
            'teacher_initialized', torch.tensor(False), persistent=True)
        self.register_buffer(
            'teacher_update_steps',
            torch.tensor(0, dtype=torch.long),
            persistent=True)
        self._freeze_teacher()

    def _teacher_student_pairs(self):
        if not self.ema_teacher_enabled:
            return []
        pairs = [(self.teacher_backbone, self.backbone)]
        if self.with_neck:
            pairs.append((self.teacher_neck, self.neck))
        pairs.append((self.teacher_bbox_head, self.bbox_head))
        return pairs

    def _freeze_teacher(self):
        for teacher_module, _ in self._teacher_student_pairs():
            teacher_module.requires_grad_(False)
            teacher_module.eval()

    @staticmethod
    def _assert_matching_state(teacher_module, student_module):
        teacher_parameters = dict(teacher_module.named_parameters())
        student_parameters = dict(student_module.named_parameters())
        if teacher_parameters.keys() != student_parameters.keys():
            raise RuntimeError('Teacher and student parameter names differ')

        teacher_buffers = dict(teacher_module.named_buffers())
        student_buffers = dict(student_module.named_buffers())
        if teacher_buffers.keys() != student_buffers.keys():
            raise RuntimeError('Teacher and student buffer names differ')
        return (teacher_parameters, student_parameters, teacher_buffers,
                student_buffers)

    @torch.no_grad()
    def initialize_teacher(self):
        if not self.ema_teacher_enabled:
            return
        if self.teacher_initialized.item():
            return
        for teacher_module, student_module in self._teacher_student_pairs():
            teacher_module.load_state_dict(student_module.state_dict(),
                                           strict=True)
        self.teacher_initialized.fill_(True)
        self.teacher_update_steps.zero_()
        self._freeze_teacher()

    @torch.no_grad()
    def update_teacher(self, momentum):
        if not self.ema_teacher_enabled or not self.teacher_initialized.item():
            return
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('EMA momentum must be in [0, 1]')

        for teacher_module, student_module in self._teacher_student_pairs():
            states = self._assert_matching_state(teacher_module,
                                                 student_module)
            teacher_parameters, student_parameters = states[:2]
            teacher_buffers, student_buffers = states[2:]
            for name, teacher_parameter in teacher_parameters.items():
                teacher_parameter.mul_(momentum).add_(
                    student_parameters[name], alpha=1.0 - momentum)
            for name, teacher_buffer in teacher_buffers.items():
                student_buffer = student_buffers[name]
                if torch.is_floating_point(teacher_buffer):
                    teacher_buffer.mul_(momentum).add_(
                        student_buffer, alpha=1.0 - momentum)
                else:
                    teacher_buffer.copy_(student_buffer)
        self.teacher_update_steps.add_(1)
        self._freeze_teacher()

    def train(self, mode=True):
        result = super().train(mode)
        if getattr(self, 'ema_teacher_enabled', False):
            self._freeze_teacher()
        return result


@MODELS.register_module()
class Point2RBoxV3EMATeacher(EMATeacherMixin, Point2RBoxV3):
    """Point2RBox-v3 with an inactive-by-default frozen EMA copy.

    The teacher is intentionally not used in ``loss`` or ``predict`` here.
    This class only establishes checkpointable EMA mechanics.
    """

    def __init__(self, *args, ema_teacher=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_ema_teacher(ema_teacher)
