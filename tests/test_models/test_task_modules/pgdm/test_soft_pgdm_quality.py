from dataclasses import dataclass

import numpy as np
import torch

from projects.point2rbox_v3_soft_pgdm.candidate_quality import (
    candidate_quality_scores, instance_quality_gate,
    soft_candidate_weights)


@dataclass(frozen=True)
class Candidate:
    mask: np.ndarray
    source_radius: float
    cluster_radii: tuple


def test_separation_penalizes_merged_same_class_instance():
    mask_clean = np.zeros((8, 8), dtype=bool)
    mask_clean[1:4, 1:4] = True
    mask_merged = mask_clean.copy()
    mask_merged[5:7, 5:7] = True
    clean = Candidate(mask_clean, 1.0, (1.0, ))
    merged = Candidate(mask_merged, 1.25, (1.25, ))
    points = np.array([[2, 2], [6, 6]], dtype=np.float32)
    labels = np.array([3, 3])

    qualities = candidate_quality_scores(
        [clean, merged], points, labels, 0, beta=0.0, delta=1.0)

    assert qualities == [1.0, 0.5]


def test_soft_weights_are_detached_and_normalized():
    probabilities = soft_candidate_weights([0.2, 0.8], 1.0, 'cpu')
    assert not probabilities.requires_grad
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0))
    assert probabilities[1] > probabilities[0]


def test_single_candidate_gate_has_no_entropy_division_by_zero():
    probabilities = torch.tensor([1.0])
    gate = instance_quality_gate(
        [0.8], probabilities, quality_threshold=0.5,
        smoothness=0.1, enabled=True)
    assert torch.isfinite(gate)
    assert gate > 0


def test_disabled_gate_preserves_full_instance_weight():
    gate = instance_quality_gate(
        [0.1, 0.1], torch.tensor([0.5, 0.5]), enabled=False)
    assert gate.item() == 1.0
