import numpy as np
import torch
import cv2

from mmrotate.models.losses.point2rbox_v2_loss import (
    VoronoiWatershedLoss, gwd_sigma_loss)
from mmrotate.models.losses.utils import filter_masks
from mmrotate.models.losses.vis import (save_debug_visualization,
                                        visualize_loss_calculation)
from mmrotate.registry import MODELS

from .candidate_pool import MultiScaleSAMCandidatePool


_SAM_MODEL_CACHE = {}


@MODELS.register_module()
class MultiScaleVoronoiWatershedLoss(VoronoiWatershedLoss):
    """Voronoi/Watershed loss with an optional multi-scale SAM branch."""

    def __init__(self,
                 enabled=True,
                 scales=(0.75, 1.0, 1.25),
                 max_masks_per_scale=3,
                 dedup_iou_thr=0.9,
                 instance_batch_size=1,
                 view_strategy='whole_resize',
                 zoom_grid_size=2,
                 crop_image_batch_size=4,
                 selector=None,
                 sam_checkpoint='./mobile_sam.pt',
                 sam_model_type='vit_t',
                 **kwargs):
        super().__init__(**kwargs)
        self.enabled = enabled
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.view_strategy = view_strategy
        self.selector = dict(
            center_weight=1 / 3,
            sam_weight=1 / 3,
            cross_view_weight=1 / 3,
            replacement_margin=0.05,
            min_mask_area=16,
            max_image_coverage=0.95,
            crop_boundary_width=3,
            crop_boundary_ratio=0.02,
            crop_boundary_penalty=0.5)
        if selector is not None:
            self.selector.update(selector)
        weight_sum = sum(self.selector[key] for key in (
            'center_weight', 'sam_weight', 'cross_view_weight'))
        if weight_sum <= 0:
            raise ValueError('selector reliability weights must sum positive')
        self.candidate_pool = MultiScaleSAMCandidatePool(
            scales=scales,
            max_masks_per_scale=max_masks_per_scale,
            dedup_iou_thr=dedup_iou_thr,
            instance_batch_size=instance_batch_size,
            view_strategy=view_strategy,
            crop_boundary_width=self.selector['crop_boundary_width'],
            zoom_grid_size=zoom_grid_size,
            crop_image_batch_size=crop_image_batch_size)

    def _build_predictor(self, device):
        try:
            from mobile_sam import SamPredictor, sam_model_registry
        except ImportError as error:
            raise ImportError(
                'Please install MobileSAM: pip install '
                'git+https://github.com/ChaoningZhang/MobileSAM.git') from error

        cache_key = (self.sam_model_type, self.sam_checkpoint, str(device))
        if cache_key not in _SAM_MODEL_CACHE:
            sam = sam_model_registry[self.sam_model_type](
                checkpoint=self.sam_checkpoint)
            sam.to(device)
            _SAM_MODEL_CACHE[cache_key] = sam
        return SamPredictor(_SAM_MODEL_CACHE[cache_key])

    @staticmethod
    def _image_to_uint8(image):
        image_min = image.min()
        image_range = image.max() - image_min
        if image_range == 0:
            normalized = torch.zeros_like(image)
        else:
            normalized = (image - image_min) / image_range * 255.0
        return normalized.permute(1, 2, 0).detach().cpu().numpy().astype(
            np.uint8)

    @staticmethod
    def _point_in_mask(mask, point):
        height, width = mask.shape
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        return 0 <= x < width and 0 <= y < height and bool(mask[y, x])

    @staticmethod
    def _mask_geometry(mask):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        (center_x, center_y), (width, height), _ = cv2.minAreaRect(contour)
        if width <= 0 or height <= 0:
            return None
        return np.asarray((center_x, center_y), dtype=np.float32), width * height

    @staticmethod
    def _cross_view_score(candidate, candidates):
        scores = []
        for other in candidates:
            if other is candidate or other.view_type == candidate.view_type:
                continue
            intersection = np.logical_and(candidate.mask, other.mask).sum()
            union = np.logical_or(candidate.mask, other.mask).sum()
            scores.append(float(intersection / union) if union else 0.0)
        return max(scores, default=0.0)

    def _candidate_reliability(self, candidate, candidates, point):
        geometry = self._mask_geometry(candidate.mask)
        if geometry is None:
            return -np.inf
        center, box_area = geometry
        distance = np.linalg.norm(center - point)
        center_score = float(np.exp(-distance / np.sqrt(max(box_area, 1e-6))))
        sam_score = float(np.clip(candidate.score, 0.0, 1.0))
        cross_score = self._cross_view_score(candidate, candidates)
        weights = self.selector
        weight_sum = (weights['center_weight'] + weights['sam_weight'] +
                      weights['cross_view_weight'])
        score = (
            weights['center_weight'] * center_score +
            weights['sam_weight'] * sam_score +
            weights['cross_view_weight'] * cross_score) / weight_sum
        if (candidate.view_type == 'zoom' and
                candidate.boundary_contact_ratio >
                weights['crop_boundary_ratio']):
            score *= weights['crop_boundary_penalty']
        return score

    def _passes_hard_filter(self, candidate, point, negative_points,
                            image_shape):
        mask = candidate.mask
        area = int(mask.sum())
        image_area = image_shape[0] * image_shape[1]
        if area < self.selector['min_mask_area']:
            return False
        if area / image_area > self.selector['max_image_coverage']:
            return False
        if not self._point_in_mask(mask, point):
            return False
        if any(self._point_in_mask(mask, negative)
               for negative in negative_points):
            return False
        return True

    def _select_asymmetric_candidate(self, instance_index, candidates,
                                     points, labels, image, image_np):
        original_candidates = [
            candidate for candidate in candidates
            if candidate.view_type == 'original'
        ]
        if not original_candidates:
            return None
        masks = [candidate.mask for candidate in original_candidates]
        scores = np.asarray([
            candidate.score for candidate in original_candidates])
        class_id = int(labels[instance_index])
        baseline_index, _, _ = filter_masks(
            image,
            masks,
            scores,
            class_id,
            image_np,
            points[instance_index],
            self.mask_filter_config,
            self.debug)
        baseline = original_candidates[baseline_index]

        prompt_indices = self.candidate_pool._prompt_indices(
            instance_index, points, labels, self.sam_sample_rules)
        negative_points = points[prompt_indices[1:]]
        valid_candidates = [
            candidate for candidate in candidates
            if candidate is not baseline and self._passes_hard_filter(
                candidate, points[instance_index], negative_points,
                image_np.shape)
        ]
        if not valid_candidates:
            return baseline

        baseline_score = self._candidate_reliability(
            baseline, candidates, points[instance_index])
        new_scores = [
            self._candidate_reliability(
                candidate, candidates, points[instance_index])
            for candidate in valid_candidates
        ]
        best_index = int(np.argmax(new_scores))
        if (new_scores[best_index] > baseline_score +
                self.selector['replacement_margin']):
            return valid_candidates[best_index]
        return baseline

    def _forward_multiscale_sam(self, mu, sigma, label, image):
        image_np = self._image_to_uint8(image)
        points = mu.detach().cpu().numpy()
        labels = label.detach().cpu().numpy()
        predictor = self._build_predictor(mu.device)
        candidates_by_instance = self.candidate_pool.generate(
            predictor,
            image_np,
            points,
            labels,
            sample_rules=self.sam_sample_rules)

        height, width = image_np.shape[:2]
        markers = torch.full(
            (height, width),
            len(mu) + 1,
            dtype=torch.int32,
            device=mu.device)
        total_loss = sigma.sum() * 0
        valid_instances = 0
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma)

        for instance_index, candidates in enumerate(candidates_by_instance):
            if not candidates:
                continue
            class_id = int(label[instance_index].item())
            if self.view_strategy == 'asymmetric':
                selected = self._select_asymmetric_candidate(
                    instance_index, candidates, points, labels, image,
                    image_np)
                if selected is None:
                    continue
                masks = [candidate.mask for candidate in candidates]
                scores = np.asarray([
                    candidate.score for candidate in candidates])
                best_index = next(
                    index for index, candidate in enumerate(candidates)
                    if candidate is selected)
                metrics, shape_metrics = [], []
            else:
                masks = [candidate.mask for candidate in candidates]
                scores = np.asarray([
                    candidate.score for candidate in candidates])
                best_index, metrics, shape_metrics = filter_masks(
                    image,
                    masks,
                    scores,
                    class_id,
                    image_np,
                    points[instance_index],
                    self.mask_filter_config,
                    self.debug)
            if self.debug and self.view_strategy != 'asymmetric':
                save_debug_visualization(
                    image, masks, scores, shape_metrics, metrics, best_index,
                    class_id, 'Multi-Scale Mask Selection')

            mask_tensor = torch.from_numpy(masks[best_index]).to(mu.device)
            markers[mask_tensor] = instance_index + 1
            coordinates = mask_tensor.nonzero()[:, (1, 0)].float()
            if len(coordinates) == 0:
                continue
            centered = coordinates - mu[instance_index]
            rotated = eigenvectors[instance_index].T.matmul(
                centered[:, :, None])[:, :, 0]
            max_x = torch.max(torch.abs(rotated[:, 0]))
            max_y = torch.max(torch.abs(rotated[:, 1]))
            target = torch.stack((max_x, max_y)).square()
            predicted_diagonal = torch.diag_embed(
                eigenvalues[instance_index])
            target_diagonal = torch.diag_embed(target)
            instance_loss = gwd_sigma_loss(
                predicted_diagonal.unsqueeze(0),
                target_diagonal.unsqueeze(0).detach(),
                reduction='mean')
            if self.debug:
                visualize_loss_calculation(
                    image, mask_tensor, mu[instance_index],
                    eigenvectors[instance_index], centered, rotated, max_x,
                    max_y, eigenvalues[instance_index], target,
                    instance_loss, instance_index, class_id)
            total_loss = total_loss + instance_loss
            valid_instances += 1

        loss = total_loss / max(1, valid_instances)
        self.vis = (markers.clone(), markers)
        return self.loss_weight * loss

    def forward(self,
                pred,
                label,
                image,
                pos_thres,
                neg_thres,
                voronoi='orientation'):
        if not self.enabled:
            return super().forward(pred, label, image, pos_thres, neg_thres,
                                   voronoi)
        mu, sigma = pred
        if len(sigma) == 0:
            return self.loss_weight * sigma.sum()
        if len(sigma) > self.sam_instance_thr:
            return super().forward(pred, label, image, pos_thres, neg_thres,
                                   voronoi)
        return self._forward_multiscale_sam(mu, sigma, label, image)
