from dataclasses import dataclass, replace
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mmrotate.models.losses.point2rbox_v2_loss import gaussian_2d


@dataclass(frozen=True)
class WatershedCandidate:
    """One watershed mask represented in the training coordinates."""

    mask: np.ndarray
    source_radius: float
    cluster_radii: Tuple[float, ...]


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 0.0


def deduplicate_candidates(
        candidates: Sequence[WatershedCandidate],
        iou_threshold: float) -> List[WatershedCandidate]:
    """Cluster masks, preferring the baseline radius representative."""
    representatives: List[WatershedCandidate] = []
    ordered = sorted(
        candidates, key=lambda item: abs(np.log(item.source_radius)))
    for candidate in ordered:
        duplicate_index = next(
            (index for index, representative in enumerate(representatives)
             if mask_iou(candidate.mask, representative.mask) > iou_threshold),
            None)
        if duplicate_index is None:
            representatives.append(candidate)
            continue
        representative = representatives[duplicate_index]
        merged_radii = tuple(sorted(set(representative.cluster_radii +
                                        candidate.cluster_radii)))
        representatives[duplicate_index] = replace(
            representative, cluster_radii=merged_radii)
    return representatives


class MultiRadiusWatershedCandidatePool:
    """Generate watershed masks from multiple Gaussian seed radii."""

    def __init__(self,
                 radius_multipliers=(0.75, 1.0, 1.25),
                 dedup_iou_thr=0.9):
        if (not radius_multipliers or
                any(radius <= 0 for radius in radius_multipliers)):
            raise ValueError('radius_multipliers must contain positive values')
        if not any(np.isclose(radius, 1.0)
                   for radius in radius_multipliers):
            raise ValueError('radius_multipliers must include 1.0')
        if not 0 <= dedup_iou_thr <= 1:
            raise ValueError('dedup_iou_thr must be in [0, 1]')
        self.radius_multipliers = tuple(
            float(radius) for radius in radius_multipliers)
        self.dedup_iou_thr = float(dedup_iou_thr)

    @staticmethod
    def _watershed_markers(mu, sigma, label, image, pos_thres, neg_thres,
                           down_sample, default_sigma, voronoi,
                           radius_multiplier):
        num_instances = len(sigma)
        height, width = image.shape[-2:]
        if height % down_sample or width % down_sample:
            raise ValueError('image size must be divisible by down_sample')
        small_height = height // down_sample
        small_width = width // down_sample
        x = torch.linspace(
            0, small_height, small_height, device=mu.device)
        y = torch.linspace(0, small_width, small_width, device=mu.device)
        xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
        distributions = mu.new_zeros(
            num_instances, small_height, small_width)
        centers = (mu.detach() / down_sample).round()
        radius_square = radius_multiplier**2

        if voronoi == 'standard':
            seed_sigma = default_sigma * radius_square
            covariance = sigma.new_tensor(
                ((seed_sigma, 0), (0, seed_sigma)))
            covariance = covariance / down_sample**2
            for index, center in enumerate(centers):
                distributions[index] = gaussian_2d(
                    xy.view(-1, 2), center[None], covariance[None]).view(
                        small_height, small_width)
        elif voronoi == 'gaussian-orientation':
            eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
            eigenvalues = eigenvalues.detach().clone()
            eigenvalues = eigenvalues / (
                eigenvalues[:, 0:1] * eigenvalues[:, 1:2]
            ).sqrt() * default_sigma * radius_square
            covariances = eigenvectors.matmul(
                torch.diag_embed(eigenvalues)).matmul(
                    eigenvectors.permute(0, 2, 1)).detach()
            covariances = covariances / down_sample**2
            for index, (center, covariance) in enumerate(
                    zip(centers, covariances)):
                distributions[index] = gaussian_2d(
                    xy.view(-1, 2), center[None], covariance[None]).view(
                        small_height, small_width)
        elif voronoi == 'gaussian-full':
            covariances = sigma.detach() * radius_square / down_sample**2
            for index, (center, covariance) in enumerate(
                    zip(centers, covariances)):
                distributions[index] = gaussian_2d(
                    xy.view(-1, 2), center[None], covariance[None]).view(
                        small_height, small_width)
        else:
            raise ValueError(f'Unsupported voronoi type: {voronoi}')

        values, regions = torch.max(distributions, 0)
        if down_sample > 1:
            regions = regions[:, None, :, None].expand(
                -1, down_sample, -1, down_sample).reshape(height, width)
            values = F.interpolate(
                values[None, None], (height, width), mode='bilinear',
                align_corners=True)[0, 0]
        classes = label[regions]
        kernel = values.new_ones((1, 1, 3, 3))
        kernel[0, 0, 1, 1] = -8
        ridges = torch.conv2d(
            regions[None].float(), kernel.float(), padding=1)[0] != 0
        regions += 1
        positive = values.new_tensor(pos_thres)
        negative = values.new_tensor(neg_thres)
        regions[values < positive[classes]] = 0
        regions[values < negative[classes]] = num_instances + 1
        regions[ridges] = num_instances + 1

        image_min = image.min()
        image_range = image.max() - image_min
        if image_range == 0:
            image_uint8 = torch.zeros_like(image)
        else:
            image_uint8 = (image - image_min) / image_range * 255
        image_uint8 = image_uint8.permute(
            1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        image_uint8 = cv2.medianBlur(image_uint8, 3)
        markers = regions.detach().cpu().numpy().astype(np.int32)
        return cv2.watershed(image_uint8, markers)

    def generate(self, image, mu, sigma, label, pos_thres, neg_thres,
                 down_sample=2, default_sigma=4096,
                 voronoi='gaussian-orientation'):
        if len(mu) == 0:
            return []
        candidates = [[] for _ in range(len(mu))]
        with torch.no_grad():
            for radius in self.radius_multipliers:
                markers = self._watershed_markers(
                    mu, sigma, label, image, pos_thres, neg_thres,
                    down_sample, default_sigma, voronoi, radius)
                for instance_index in range(len(mu)):
                    mask = markers == instance_index + 1
                    if not mask.any():
                        continue
                    candidates[instance_index].append(
                        WatershedCandidate(
                            mask=mask,
                            source_radius=radius,
                            cluster_radii=(radius, )))
        return [
            deduplicate_candidates(instance_candidates,
                                   self.dedup_iou_thr)
            for instance_candidates in candidates
        ]
