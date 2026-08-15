from unittest.mock import patch

import torch

from mmrotate.models.losses.point2rbox_v2_loss import VoronoiWatershedLoss
from projects.point2rbox_v3_multiradius_watershed import MultiRadiusPGDMLoss


def _inputs(num_instances):
    mu = torch.stack([
        torch.tensor([12.0 + 8 * index, 16.0])
        for index in range(num_instances)
    ])
    sigma = torch.eye(2).mul(16).repeat(num_instances, 1, 1)
    label = torch.zeros(num_instances, dtype=torch.long)
    image = torch.zeros((3, 64, 64))
    return (mu, sigma), label, image


def test_sam_and_multiradius_pool_switches_are_independent():
    loss_module = MultiRadiusPGDMLoss(
        sam_candidate_pool_enabled=True,
        watershed_candidate_pool_enabled=True,
        sam_instance_thr=1)
    sparse_pred, sparse_label, image = _inputs(1)
    dense_pred, dense_label, _ = _inputs(2)
    sparse_expected = torch.tensor(1.0)
    dense_expected = torch.tensor(2.0)
    with patch.object(
            loss_module, '_forward_multiscale_sam',
            return_value=sparse_expected) as sam_forward, patch.object(
                loss_module, '_forward_multiradius_watershed',
                return_value=dense_expected) as watershed_forward:
        sparse_actual = loss_module(
            sparse_pred, sparse_label, image, [0.9], [0.1], 'standard')
        dense_actual = loss_module(
            dense_pred, dense_label, image, [0.9], [0.1], 'standard')
    assert sparse_actual is sparse_expected
    assert dense_actual is dense_expected
    sam_forward.assert_called_once()
    watershed_forward.assert_called_once()


def test_disabled_candidate_pools_delegate_to_original_pgdm():
    loss_module = MultiRadiusPGDMLoss(
        sam_candidate_pool_enabled=False,
        watershed_candidate_pool_enabled=False,
        sam_instance_thr=1)
    pred, label, image = _inputs(2)
    expected = torch.tensor(3.0)
    with patch.object(
            VoronoiWatershedLoss, 'forward',
            return_value=expected) as parent_forward:
        actual = loss_module(
            pred, label, image, [0.9], [0.1], 'standard')
    assert actual is expected
    parent_forward.assert_called_once()


def test_single_radius_watershed_matches_original_loss_and_has_gradients():
    pred, label, image = _inputs(2)
    image[:, 8:56, 8:56] = 1
    pred = (pred[0], pred[1].requires_grad_())
    kwargs = dict(
        loss_weight=1.0,
        down_sample=2,
        topk=0.95,
        default_sigma=4096,
        sam_instance_thr=0)
    baseline = VoronoiWatershedLoss(**kwargs)
    multiradius = MultiRadiusPGDMLoss(
        **kwargs,
        watershed_candidate_pool_enabled=True,
        watershed_radius_multipliers=(1.0, ))
    baseline_loss = baseline(
        pred, label, image, [0.9], [0.1], 'standard')
    multiradius_loss = multiradius(
        pred, label, image, [0.9], [0.1], 'standard')
    assert torch.allclose(multiradius_loss, baseline_loss, atol=1e-6)
    multiradius_loss.backward()
    assert pred[1].grad is not None
    assert torch.isfinite(pred[1].grad).all()
