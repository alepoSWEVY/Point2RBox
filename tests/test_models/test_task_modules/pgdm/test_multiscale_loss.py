from unittest.mock import patch

import torch

from mmrotate.models.losses.point2rbox_v2_loss import VoronoiWatershedLoss
from projects.point2rbox_v3_multiscale.multiscale_voronoi_loss import (
    MultiScaleVoronoiWatershedLoss)


def test_disabled_component_delegates_to_parent():
    loss_module = MultiScaleVoronoiWatershedLoss(enabled=False)
    expected = torch.tensor(2.5)
    pred = (torch.zeros((1, 2)), torch.eye(2).unsqueeze(0))
    label = torch.zeros(1, dtype=torch.long)
    image = torch.zeros((3, 16, 16))
    with patch.object(VoronoiWatershedLoss, 'forward', return_value=expected) \
            as parent_forward:
        actual = loss_module(pred, label, image, [0.9], [0.1])
    assert actual is expected
    parent_forward.assert_called_once()


def test_enabled_component_uses_candidate_pool_for_sparse_instances():
    loss_module = MultiScaleVoronoiWatershedLoss(
        enabled=True, sam_instance_thr=4)
    pred = (torch.zeros((1, 2)), torch.eye(2).unsqueeze(0))
    label = torch.zeros(1, dtype=torch.long)
    image = torch.zeros((3, 16, 16))
    expected = torch.tensor(1.25)
    with patch.object(
            loss_module, '_forward_multiscale_sam', return_value=expected
    ) as multiscale_forward:
        actual = loss_module(pred, label, image, [0.9], [0.1])
    assert actual is expected
    multiscale_forward.assert_called_once()
