import numpy as np
import torch

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
                 sam_checkpoint='./mobile_sam.pt',
                 sam_model_type='vit_t',
                 **kwargs):
        super().__init__(**kwargs)
        self.enabled = enabled
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.candidate_pool = MultiScaleSAMCandidatePool(
            scales=scales,
            max_masks_per_scale=max_masks_per_scale,
            dedup_iou_thr=dedup_iou_thr)

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
            masks = [candidate.mask for candidate in candidates]
            scores = np.asarray([candidate.score for candidate in candidates])
            class_id = int(label[instance_index].item())
            best_index, metrics, shape_metrics = filter_masks(
                image,
                masks,
                scores,
                class_id,
                image_np,
                points[instance_index],
                self.mask_filter_config,
                self.debug)
            if self.debug:
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
