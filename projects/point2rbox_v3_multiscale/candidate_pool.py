from dataclasses import dataclass, replace
from collections import defaultdict
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
    view_type: str = 'whole_resize'
    boundary_contact_ratio: float = 0.0


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


@dataclass(frozen=True)
class CropTransform:
    """Coordinate transform for one point-centred, possibly padded crop."""

    original_size: Tuple[int, int]
    crop_size: Tuple[int, int]
    x0: int
    y0: int

    @classmethod
    def from_point(cls, image_shape: Sequence[int], point: np.ndarray,
                   zoom_scale: float):
        height, width = image_shape[:2]
        crop_height = max(1, int(round(height / zoom_scale)))
        crop_width = max(1, int(round(width / zoom_scale)))
        x0 = int(round(float(point[0]) - crop_width / 2))
        y0 = int(round(float(point[1]) - crop_height / 2))
        return cls((height, width), (crop_height, crop_width), x0, y0)

    def map_points(self, points: np.ndarray) -> np.ndarray:
        mapped = np.asarray(points, dtype=np.float32).copy()
        mapped[..., 0] -= self.x0
        mapped[..., 1] -= self.y0
        return mapped

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        crop_height, crop_width = self.crop_size
        mapped = self.map_points(points)
        return ((mapped[..., 0] >= 0) & (mapped[..., 0] < crop_width) &
                (mapped[..., 1] >= 0) & (mapped[..., 1] < crop_height))

    def extract_image(self, image: np.ndarray) -> np.ndarray:
        height, width = self.original_size
        crop_height, crop_width = self.crop_size
        left = max(0, -self.x0)
        top = max(0, -self.y0)
        right = max(0, self.x0 + crop_width - width)
        bottom = max(0, self.y0 + crop_height - height)
        padded = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        start_x = self.x0 + left
        start_y = self.y0 + top
        return padded[start_y:start_y + crop_height,
                      start_x:start_x + crop_width]

    def restore_mask(self, mask: np.ndarray) -> np.ndarray:
        height, width = self.original_size
        crop_height, crop_width = self.crop_size
        restored = np.zeros((height, width), dtype=bool)
        original_x0 = max(0, self.x0)
        original_y0 = max(0, self.y0)
        original_x1 = min(width, self.x0 + crop_width)
        original_y1 = min(height, self.y0 + crop_height)
        if original_x0 >= original_x1 or original_y0 >= original_y1:
            return restored
        crop_x0 = original_x0 - self.x0
        crop_y0 = original_y0 - self.y0
        crop_x1 = crop_x0 + original_x1 - original_x0
        crop_y1 = crop_y0 + original_y1 - original_y0
        restored[original_y0:original_y1, original_x0:original_x1] = \
            mask[crop_y0:crop_y1, crop_x0:crop_x1]
        return restored


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
                 dedup_iou_thr=0.9,
                 instance_batch_size=1,
                 view_strategy='whole_resize',
                 crop_boundary_width=3,
                 zoom_grid_size=2,
                 crop_image_batch_size=4):
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError('scales must contain positive values')
        if not any(np.isclose(scale, 1.0) for scale in scales):
            raise ValueError('scales must include the legacy 1.0 view')
        if max_masks_per_scale <= 0:
            raise ValueError('max_masks_per_scale must be positive')
        if not 0 <= dedup_iou_thr <= 1:
            raise ValueError('dedup_iou_thr must be in [0, 1]')
        if instance_batch_size <= 0:
            raise ValueError('instance_batch_size must be positive')
        if view_strategy not in ('whole_resize', 'asymmetric'):
            raise ValueError(
                'view_strategy must be "whole_resize" or "asymmetric"')
        if crop_boundary_width < 0:
            raise ValueError('crop_boundary_width must be non-negative')
        if zoom_grid_size <= 0:
            raise ValueError('zoom_grid_size must be positive')
        if crop_image_batch_size <= 0:
            raise ValueError('crop_image_batch_size must be positive')
        self.scales = tuple(float(scale) for scale in scales)
        self.max_masks_per_scale = int(max_masks_per_scale)
        self.dedup_iou_thr = float(dedup_iou_thr)
        self.instance_batch_size = int(instance_batch_size)
        self.view_strategy = view_strategy
        self.crop_boundary_width = int(crop_boundary_width)
        self.zoom_grid_size = int(zoom_grid_size)
        self.crop_image_batch_size = int(crop_image_batch_size)

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

    def _predict_current_view(self, predictor, mapped_points, prompt_indices,
                              instance_indices=None):
        predictions = {}
        if instance_indices is None:
            instance_indices = list(range(len(prompt_indices)))
        if self.instance_batch_size == 1:
            for instance_index in instance_indices:
                indices = prompt_indices[instance_index]
                point_coords = mapped_points[indices]
                point_labels = np.zeros(len(indices), dtype=np.int32)
                point_labels[0] = 1
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=True)
                predictions[instance_index] = (masks, scores)
            return predictions

        import torch
        groups = defaultdict(list)
        for instance_index in instance_indices:
            indices = prompt_indices[instance_index]
            groups[len(indices)].append((instance_index, indices))
        for group in groups.values():
            for offset in range(0, len(group), self.instance_batch_size):
                chunk = group[offset:offset + self.instance_batch_size]
                coords_np = np.stack([
                    mapped_points[indices] for _, indices in chunk
                ])
                labels_np = np.zeros(coords_np.shape[:2], dtype=np.int32)
                labels_np[:, 0] = 1
                coords_np = predictor.transform.apply_coords(
                    coords_np, predictor.original_size)
                coords = torch.as_tensor(
                    coords_np, dtype=torch.float, device=predictor.device)
                point_labels = torch.as_tensor(
                    labels_np, dtype=torch.int, device=predictor.device)
                masks, scores, _ = predictor.predict_torch(
                    point_coords=coords,
                    point_labels=point_labels,
                    boxes=None,
                    mask_input=None,
                    multimask_output=True)
                masks = masks.detach().cpu().numpy()
                scores = scores.detach().cpu().numpy()
                for batch_index, (instance_index, _) in enumerate(chunk):
                    predictions[instance_index] = (
                        masks[batch_index], scores[batch_index])
        return predictions

    def _predict_shared_view(self, predictor, image, mapped_points,
                             prompt_indices):
        predictor.set_image(image)
        return self._predict_current_view(
            predictor, mapped_points, prompt_indices)

    def _encode_image_batch(self, predictor, images):
        """Encode equal-shaped crop views in configurable image batches."""
        import torch
        encoded = []
        for offset in range(0, len(images), self.crop_image_batch_size):
            chunk = images[offset:offset + self.crop_image_batch_size]
            transformed = [predictor.transform.apply_image(image)
                           for image in chunk]
            input_sizes = [item.shape[:2] for item in transformed]
            if len(set(input_sizes)) != 1:
                raise ValueError('crop views in one image batch must share size')
            tensors = [torch.as_tensor(item, device=predictor.device)
                       .permute(2, 0, 1).contiguous()
                       for item in transformed]
            batch = torch.stack(tensors)
            with torch.no_grad():
                features = predictor.model.image_encoder(
                    predictor.model.preprocess(batch))
            for index, image in enumerate(chunk):
                encoded.append((features[index:index + 1], input_sizes[index],
                                image.shape[:2]))
        return encoded

    @staticmethod
    def _activate_encoded_view(predictor, encoded):
        feature, input_size, original_size = encoded
        predictor.reset_image()
        predictor.features = feature
        predictor.input_size = tuple(input_size)
        predictor.original_size = tuple(original_size)
        predictor.is_image_set = True

    def _append_candidates(self, candidates, predictions, transform, scale,
                           view_type, instance_indices=None):
        if instance_indices is None:
            instance_indices = range(len(candidates))
        for instance_index in instance_indices:
            masks, scores = predictions[instance_index]
            limit = min(len(masks), self.max_masks_per_scale)
            for source_index in range(limit):
                canonical_mask = transform.restore_mask(masks[source_index])
                canonical_mask = _postprocess_mask(canonical_mask)
                if not canonical_mask.any():
                    continue
                candidates[instance_index].append(Candidate(
                    mask=canonical_mask,
                    score=float(scores[source_index]),
                    source_scale=scale,
                    source_index=source_index,
                    cluster_scales=(scale, ),
                    view_type=view_type))

    @staticmethod
    def _boundary_contact_ratio(mask, width):
        if width <= 0 or not mask.any():
            return 0.0
        boundary = np.zeros_like(mask, dtype=bool)
        boundary[:width] = True
        boundary[-width:] = True
        boundary[:, :width] = True
        boundary[:, -width:] = True
        return float(np.logical_and(mask, boundary).sum() / mask.sum())

    def _zoom_grid_transforms(self, image_shape, zoom_scale):
        height, width = image_shape[:2]
        crop_height = max(1, int(round(height / zoom_scale)))
        crop_width = max(1, int(round(width / zoom_scale)))

        def starts(length, crop_length):
            maximum = max(0, length - crop_length)
            return sorted(set(np.linspace(
                0, maximum, self.zoom_grid_size).round().astype(int)))

        return [
            CropTransform((height, width), (crop_height, crop_width), x0, y0)
            for y0 in starts(height, crop_height)
            for x0 in starts(width, crop_width)
        ]

    @staticmethod
    def _assign_points_to_grid(points, transforms):
        assignments = [[] for _ in transforms]
        for instance_index, point in enumerate(points):
            choices = []
            for transform_index, transform in enumerate(transforms):
                mapped = transform.map_points(point)
                crop_height, crop_width = transform.crop_size
                x, y = float(mapped[0]), float(mapped[1])
                if 0 <= x < crop_width and 0 <= y < crop_height:
                    margin = min(x, y, crop_width - 1 - x,
                                 crop_height - 1 - y)
                    choices.append((margin, transform_index))
            if not choices:
                raise RuntimeError('zoom grid does not cover every point')
            assignments[max(choices)[1]].append(instance_index)
        return assignments

    def _generate_asymmetric(self, predictor, image, points, labels,
                             sample_rules):
        candidates = [[] for _ in range(len(points))]
        prompt_indices = [
            self._prompt_indices(index, points, labels, sample_rules)
            for index in range(len(points))
        ]
        original_height, original_width = image.shape[:2]
        identity = ScaleTransform.from_scale(image.shape, 1.0)

        texture_scale = min(self.scales)
        texture_height = max(1, int(round(original_height * texture_scale)))
        texture_width = max(1, int(round(original_width * texture_scale)))
        texture = cv2.resize(
            image, (texture_width, texture_height), interpolation=cv2.INTER_AREA)
        texture = cv2.resize(
            texture, (original_width, original_height),
            interpolation=cv2.INTER_LINEAR)
        predictions = self._predict_shared_view(
            predictor, texture, points, prompt_indices)
        self._append_candidates(
            candidates, predictions, identity, texture_scale, 'texture')

        predictions = self._predict_shared_view(
            predictor, image, points, prompt_indices)
        self._append_candidates(
            candidates, predictions, identity, 1.0, 'original')

        zoom_scale = max(self.scales)
        transforms = self._zoom_grid_transforms(image.shape, zoom_scale)
        assignments = self._assign_points_to_grid(points, transforms)
        crops = [transform.extract_image(image) for transform in transforms]
        encoded_views = self._encode_image_batch(predictor, crops)
        for transform, crop, encoded, instance_indices in zip(
                transforms, crops, encoded_views, assignments):
            if not instance_indices:
                continue
            self._activate_encoded_view(predictor, encoded)
            mapped_points = transform.map_points(points)
            inside = transform.contains_points(points)
            local_prompts = list(prompt_indices)
            for instance_index in instance_indices:
                local_prompts[instance_index] = [
                    index for index in prompt_indices[instance_index]
                    if inside[index]]
            predictions = self._predict_current_view(
                predictor, mapped_points, local_prompts, instance_indices)
            for instance_index in instance_indices:
                masks, scores = predictions[instance_index]
                limit = min(len(masks), self.max_masks_per_scale)
                for source_index in range(limit):
                    view_mask = masks[source_index].astype(bool)
                    boundary_ratio = self._boundary_contact_ratio(
                        view_mask, self.crop_boundary_width)
                    canonical_mask = transform.restore_mask(view_mask)
                    canonical_mask = _postprocess_mask(canonical_mask)
                    if not canonical_mask.any():
                        continue
                    candidates[instance_index].append(Candidate(
                        mask=canonical_mask,
                        score=float(scores[source_index]),
                        source_scale=zoom_scale,
                        source_index=source_index,
                        cluster_scales=(zoom_scale, ),
                        view_type='zoom',
                        boundary_contact_ratio=boundary_ratio))
        return candidates

    def generate(self, predictor, image: np.ndarray, points: np.ndarray,
                 labels: np.ndarray, sample_rules=None):
        points = np.asarray(points, dtype=np.float32)
        labels = np.asarray(labels)
        if self.view_strategy == 'asymmetric':
            return self._generate_asymmetric(
                predictor, image, points, labels, sample_rules)

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
            predictions = self._predict_shared_view(
                predictor, scaled_image, scaled_points, prompt_indices)
            self._append_candidates(
                candidates, predictions, transform, scale, 'whole_resize')
        return [
            deduplicate_candidates(instance_candidates, self.dedup_iou_thr)
            for instance_candidates in candidates
        ]
