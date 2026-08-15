from unittest.mock import patch

import numpy as np
import torch

from projects.point2rbox_v3_multiradius_watershed.candidate_pool import (
    MultiRadiusWatershedCandidatePool, WatershedCandidate,
    deduplicate_candidates)


def test_multiradius_dedup_prefers_baseline_radius():
    mask = np.ones((8, 8), dtype=bool)
    candidates = [
        WatershedCandidate(mask, 0.75, (0.75, )),
        WatershedCandidate(mask.copy(), 1.0, (1.0, )),
        WatershedCandidate(mask.copy(), 1.25, (1.25, )),
    ]
    result = deduplicate_candidates(candidates, 0.9)
    assert len(result) == 1
    assert result[0].source_radius == 1.0
    assert result[0].cluster_radii == (0.75, 1.0, 1.25)


def test_multiradius_pool_generates_all_radii_without_resizing_image():
    pool = MultiRadiusWatershedCandidatePool(dedup_iou_thr=1.0)
    image = torch.zeros((3, 64, 80))
    mu = torch.tensor([[40.0, 32.0]])
    sigma = torch.eye(2).mul(16).unsqueeze(0)
    label = torch.zeros(1, dtype=torch.long)
    seen = []

    def fake_markers(mu, sigma, label, image, *args):
        radius = args[-1]
        seen.append((image.shape[-2:], radius))
        markers = np.full(image.shape[-2:], 2, dtype=np.int32)
        half_size = int(round(radius * 4))
        markers[32 - half_size:33 + half_size,
                40 - half_size:41 + half_size] = 1
        return markers

    with patch.object(pool, '_watershed_markers', side_effect=fake_markers):
        result = pool.generate(
            image, mu, sigma, label, [0.9], [0.1], voronoi='standard')

    assert seen == [((64, 80), 0.75), ((64, 80), 1.0),
                    ((64, 80), 1.25)]
    assert result[0]
    assert all(candidate.mask.shape == (64, 80)
               for candidate in result[0])


def test_multiradius_pool_accepts_empty_instances():
    pool = MultiRadiusWatershedCandidatePool()
    result = pool.generate(
        torch.zeros((3, 64, 64)),
        torch.zeros((0, 2)),
        torch.zeros((0, 2, 2)),
        torch.zeros(0, dtype=torch.long),
        [0.9],
        [0.1],
        voronoi='standard')
    assert result == []
