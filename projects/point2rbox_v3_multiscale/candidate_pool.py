from dataclasses import dataclass, replace
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    """A SAM candidate represented in the current training coordinates."""

    mask: np.ndarray
    score: float
    source_scale: float
    source_index: int
    cluster_scales: Tuple[float, ...]


@dataclass(frozen=True)
class ScaleTransform:
    """Coordinate transform between an image and one scaled SAM view."""

    original_size: Tuple[int, int]
    scaled_size: Tuple[int, int]
    scale_x: float
    scale_y: float

    @classmethod
    def from_scale(cls, image_shape: Sequence[int], scale: float):
        height, width = image_shape[:2]
        scaled_height = max(1, int(round(height * scale)))
        scaled_width = max(1, int(round(width * scale)))
        return cls(
            original_size=(height, width),
            scaled_size=(scaled_height, scaled_width),
            scale_x=scaled_width / width,
            scale_y=scaled_height / height)

    def map_points(self, points: np.ndarray) -> np.ndarray:
        mapped = np.asarray(points, dtype=np.float32).copy()
        mapped[..., 0] *= self.scale_x
        mapped[..., 1] *= self.scale_y
        return mapped

    def restore_mask(self, mask: np.ndarray) -> np.ndarray:
        height, width = self.original_size
        return cv2.resize(
            mask.astype(np.uint8), (width, height),
            interpolation=cv2.INTER_NEAREST).astype(bool)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 0.0


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                              kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if num_labels <= 1:
        return opened.astype(bool)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest_label


def deduplicate_candidates(candidates: Sequence[Candidate],
                           iou_threshold: float) -> List[Candidate]:
    """Greedily cluster masks, retaining the highest-SAM-score member."""
    representatives: List[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score,
                            reverse=True):
        duplicate_index = next(
            (index for index, representative in enumerate(representatives)
             if mask_iou(candidate.mask, representative.mask) > iou_threshold),
            None)
        if duplicate_index is None:
            representatives.append(candidate)
            continue
        representative = representatives[duplicate_index]
        merged_scales = tuple(sorted(set(representative.cluster_scales +
                                         candidate.cluster_scales)))
        representatives[duplicate_index] = replace(
            representative, cluster_scales=merged_scales)
    return representatives


class MultiScaleSAMCandidatePool:
    """Generate canonical SAM masks for multiple external image scales."""

    def __init__(self,
                 scales=(0.75, 1.0, 1.25),
                 max_masks_per_scale=3,
                 dedup_iou_thr=0.9):
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError('scales must contain positive values')
        if not any(np.isclose(scale, 1.0) for scale in scales):
            raise ValueError('scales must include the legacy 1.0 view')
        if max_masks_per_scale <= 0:
            raise ValueError('max_masks_per_scale must be positive')
        if not 0 <= dedup_iou_thr <= 1:
            raise ValueError('dedup_iou_thr must be in [0, 1]')
        self.scales = tuple(float(scale) for scale in scales)
        self.max_masks_per_scale = int(max_masks_per_scale)
        self.dedup_iou_thr = float(dedup_iou_thr)

    @staticmethod
    def _prompt_indices(instance_index, points, labels, sample_rules):
        indices = [instance_index]
        for other_index in range(len(points)):
            if other_index == instance_index:
                continue
            skip = False
            if sample_rules is not None:
                distance = np.linalg.norm(points[instance_index] -
                                          points[other_index])
                for class_a, class_b, distance_thr in sample_rules.get(
                        'filter_pairs', []):
                    label_a = int(labels[instance_index])
                    label_b = int(labels[other_index])
                    matching_pair = ((label_a == class_a and
                                      label_b == class_b) or
                                     (label_a == class_b and
                                      label_b == class_a))
                    if matching_pair and distance < distance_thr:
                        skip = True
                        break
            if not skip:
                indices.append(other_index)
        return indices

    def generate(self, predictor, image: np.ndarray, points: np.ndarray,
                 labels: np.ndarray, sample_rules=None):
        points = np.asarray(points, dtype=np.float32)
        labels = np.asarray(labels)
        candidates = [[] for _ in range(len(points))]
        prompt_indices = [
            self._prompt_indices(index, points, labels, sample_rules)
            for index in range(len(points))
        ]

        for scale in self.scales:
            transform = ScaleTransform.from_scale(image.shape, scale)
            scaled_height, scaled_width = transform.scaled_size
            scaled_image = cv2.resize(
                image, (scaled_width, scaled_height),
                interpolation=cv2.INTER_LINEAR)
            scaled_points = transform.map_points(points)
            predictor.set_image(scaled_image)

            for instance_index, indices in enumerate(prompt_indices):
                point_coords = scaled_points[indices]
                point_labels = np.zeros(len(indices), dtype=np.int32)
                point_labels[0] = 1
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=True)
                limit = min(len(masks), self.max_masks_per_scale)
                for source_index in range(limit):
                    canonical_mask = transform.restore_mask(masks[source_index])
                    canonical_mask = _postprocess_mask(canonical_mask)
                    if not canonical_mask.any():
                        continue
                    candidates[instance_index].append(
                        Candidate(
                            mask=canonical_mask,
                            score=float(scores[source_index]),
                            source_scale=scale,
                            source_index=source_index,
                            cluster_scales=(scale, )))

        return [
            deduplicate_candidates(instance_candidates, self.dedup_iou_thr)
            for instance_candidates in candidates
        ]
